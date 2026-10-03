"""This module implements the parallel evaluator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

import numpy as np

from ropt._logging import get_logger
from ropt.components.executors import ExecutorFailure, WorkItem, WorkNotRun
from ropt.evaluation import EvaluationBatchContext, EvaluationBatchResult
from ropt.exceptions import ExecutionError, OptimizerStop, WorkflowError

from ._common import _active_evaluations, _build_metadata, _scatter_result
from ._counter import BatchIdCounter
from .base import EvaluationFunctionCallback, EvaluationFunctionResult, Evaluator

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from ropt.components.concurrency import AbortSignal
    from ropt.components.executors import Executor

_logger = get_logger(__name__)


class ParallelEvaluator(Evaluator):
    """An evaluator that dispatches evaluations to an executor.

    Sends each active row of the evaluation batch to the executor as its own
    work item and returns once every result is back. How many of them travel to
    a worker together follows `bundle_size`.

    See [Parallel Evaluation](../advanced/parallel.md#parallelevaluator) for
    details.
    """

    def __init__(
        self,
        *,
        function: EvaluationFunctionCallback,
        executor: Executor,
        batch_id_callback: Callable[[], int] | None = None,
        bundle_size: int | None = None,
        abort_signal: AbortSignal | None = None,
    ) -> None:
        """Initialize the ParallelEvaluator.

        Args:
            function:          The function used for objectives and constraints.
            executor:          The executor to dispatch evaluations to.
            batch_id_callback: Callable that returns the next batch ID each time it is called.
            bundle_size:       Evaluations per worker task, `None` for the executor's own.
            abort_signal:      An optional signal that abandons a running batch.
        """
        super().__init__()
        self._function = function
        self._executor = executor
        self._bundle_size = bundle_size
        self._abort_signal = abort_signal
        self._batch_id_callback = (
            batch_id_callback if batch_id_callback is not None else BatchIdCounter()
        )

    @override
    def _eval(
        self, variables: NDArray[np.float64], evaluator_context: EvaluationBatchContext
    ) -> EvaluationBatchResult:
        """Evaluate all objective and constraints.

        An infrastructure failure raises
        [`ExecutionError`][ropt.exceptions.ExecutionError]; a user-code
        exception is re-raised unchanged, leaving the executor usable. Raises
        [`ExecutorStopped`][ropt.exceptions.ExecutorStopped] if the executor
        could no longer run the work. See
        [error handling](../advanced/parallel.md#error-handling) for the full
        contract.

        Args:
            variables:         The matrix of variables to evaluate.
            evaluator_context: The evaluation context.

        Returns:
            The result of calling the wrapped evaluator function.

        Raises:
            OptimizerStop: If the stop signal abandoned the batch.
        """
        batch_id = self._batch_id_callback()

        no = evaluator_context.context.objectives.weights.size
        nc = (
            0
            if evaluator_context.context.nonlinear_constraints is None
            else evaluator_context.context.nonlinear_constraints.lower_bounds.size
        )

        results = np.zeros((variables.shape[0], no + nc), dtype=np.float64)
        metadata: dict[str, dict[int, Any]] = {}

        active = list(_active_evaluations(evaluator_context, batch_id))
        _logger.debug("Dispatching %d work item(s) to executor", len(active))
        # Only the function and one row's arguments cross to the worker.
        values = self._executor.run(
            [
                WorkItem(
                    function=self._function, args=(variables[eval_idx, :], run_context)
                )
                for eval_idx, run_context in active
            ],
            bundle_size=self._bundle_size,
            abort_signal=self._abort_signal,
        )
        # Work is only left unrun when the signal abandoned the batch, so the
        # rows that did come back describe a batch nobody is waiting for.
        if self._abort_signal is not None and any(
            isinstance(value, WorkNotRun) for value in values
        ):
            raise OptimizerStop(self._abort_signal.exit_code)
        for (eval_idx, _), value in zip(active, values, strict=True):
            _handle_result(eval_idx, value, results, metadata, no)

        return EvaluationBatchResult(
            batch_id=batch_id,
            objectives=results[:, :no],
            constraints=results[:, no:] if nc > 0 else None,
            metadata=_build_metadata(metadata, variables.shape[0]),
        )


def _handle_result(
    eval_idx: int,
    value: Any,  # ruff: ignore[any-type]
    results: NDArray[np.float64],
    metadata: dict[str, dict[int, Any]],
    objective_count: int,
) -> None:
    if isinstance(value, (ExecutorFailure, WorkNotRun)):
        msg = f"An evaluation could not be run: {value.message}"
        raise ExecutionError(msg)
    if not isinstance(value, EvaluationFunctionResult):
        msg = (
            "The evaluation function must return EvaluationFunctionResult "
            f"objects, got {type(value).__name__}."
        )
        raise WorkflowError(msg)
    _scatter_result(eval_idx, value, results, metadata, objective_count)
