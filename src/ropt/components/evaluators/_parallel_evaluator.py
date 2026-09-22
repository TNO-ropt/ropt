"""This module implements the parallel evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np

from ropt._logging import get_logger
from ropt.components.executors import (
    Executor,
    ExecutorFailure,
    Submission,
    WorkItem,
)
from ropt.evaluation import EvaluationBatchContext, EvaluationBatchResult
from ropt.exceptions import ExecutionError, WorkflowError

from ._common import _active_evaluations, _build_metadata, _scatter_result
from ._counter import BatchIdCounter
from .base import EvaluationFunctionCallback, EvaluationFunctionResult, Evaluator

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

_logger = get_logger(__name__)


@dataclass(kw_only=True)
class _EvaluationItem(WorkItem):
    # Results arrive as they finish, so each item carries the row it fills.
    eval_idx: int


class ParallelEvaluator(Evaluator):
    """An evaluator that dispatches tasks to an executor via asyncio.

    Submits each active row of the evaluation batch as its own work item and
    collects the results. How many of them travel to a worker together follows
    `bundle_size`.

    See [Parallel Evaluation](../advanced/parallel.md#parallelevaluator) for
    details on how this integrates with the asyncio event loop.
    """

    def __init__(
        self,
        *,
        function: EvaluationFunctionCallback,
        executor: Executor,
        batch_id_callback: Callable[[], int] | None = None,
        bundle_size: int = 1,
    ) -> None:
        """Initialize the ParallelEvaluator.

        Args:
            function:          The function used for objectives and constraints.
            executor:          The executor to dispatch tasks to.
            batch_id_callback: Callable that returns the next batch ID each time it is called.
            bundle_size:       Evaluations per worker task, `0` for a whole batch.

        Raises:
            ValueError: If `bundle_size` is negative.
        """
        super().__init__()
        if bundle_size < 0:
            msg = f"bundle_size must be >= 0, got {bundle_size}"
            raise ValueError(msg)
        self._function = function
        self._executor = executor
        self._bundle_size = bundle_size
        self._batch_id_callback = (
            batch_id_callback if batch_id_callback is not None else BatchIdCounter()
        )

    def _eval(
        self, variables: NDArray[np.float64], evaluator_context: EvaluationBatchContext
    ) -> EvaluationBatchResult:
        """Evaluate all objective and constraints.

        An infrastructure failure raises
        [`ExecutionError`][ropt.exceptions.ExecutionError]; a user-code
        exception is re-raised unchanged, leaving the executor running. Raises
        [`ExecutorStopped`][ropt.exceptions.ExecutorStopped] if the executor
        stopped before every result arrived. See
        [error handling](../advanced/parallel.md#error-handling) for the full
        contract.

        Args:
            variables:         The matrix of variables to evaluate.
            evaluator_context: The evaluation context.

        Returns:
            The result of calling the wrapped evaluator function.

        Raises:
            WorkflowError: If called on the executor's own event loop thread.
        """
        # This call blocks until every work item is back, so running it on the
        # loop would starve the very tasks it waits for.
        if self._executor.on_worker_loop():
            msg = (
                "A compute step must run in a thread, for example with "
                "asyncio.to_thread."
            )
            raise WorkflowError(msg)
        batch_id = self._batch_id_callback()

        no = evaluator_context.context.objectives.weights.size
        nc = (
            0
            if evaluator_context.context.nonlinear_constraints is None
            else evaluator_context.context.nonlinear_constraints.lower_bounds.size
        )

        results = np.zeros((variables.shape[0], no + nc), dtype=np.float64)
        metadata: dict[str, dict[int, Any]] = {}

        # Only the function and one row's arguments cross to the worker; the
        # delivery channel stays here, on the submission.
        submission = Submission(
            [
                _EvaluationItem(
                    function=self._function,
                    args=(variables[eval_idx, :], function_context),
                    eval_idx=eval_idx,
                )
                for eval_idx, function_context in _active_evaluations(
                    evaluator_context, batch_id
                )
            ],
            bundle_size=self._bundle_size,
        )
        _logger.debug(
            "Dispatching %d work item(s) to executor", len(submission.work_items)
        )
        self._executor.submit(submission)
        # Blocks until every work item is delivered; a user-code exception from
        # a worker is re-raised here, unchanged, ending this evaluation.
        submission.collect(
            partial(
                _handle_result,
                results=results,
                metadata=metadata,
                objective_count=no,
            ),
        )

        return EvaluationBatchResult(
            batch_id=batch_id,
            objectives=results[:, :no],
            constraints=results[:, no:] if nc > 0 else None,
            metadata=_build_metadata(metadata, variables.shape[0]),
        )


def _handle_result(
    work_item: WorkItem,
    results: NDArray[np.float64],
    metadata: dict[str, dict[int, Any]],
    objective_count: int,
) -> None:
    assert isinstance(work_item, _EvaluationItem)
    if isinstance(work_item.result, ExecutorFailure):
        msg = f"An evaluation could not be run: {work_item.result.message}"
        raise ExecutionError(msg)
    if not isinstance(work_item.result, EvaluationFunctionResult):
        msg = (
            "The evaluation function must return EvaluationFunctionResult "
            f"objects, got {type(work_item.result).__name__}."
        )
        raise WorkflowError(msg)
    _scatter_result(
        work_item.eval_idx,
        work_item.result,
        results,
        metadata,
        objective_count,
    )
