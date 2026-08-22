"""This module implements the default function evaluator."""

from __future__ import annotations

from functools import partial
from itertools import starmap
from typing import TYPE_CHECKING, Any

import numpy as np

from ropt._logging import get_logger
from ropt.components.executors import Executor, Submission, WorkItem
from ropt.evaluation import EvaluationBatchContext, EvaluationBatchResult
from ropt.exceptions import ExecutorFailure, WorkflowError

from ._common import _active_evaluations, _scatter_result
from ._counter import BatchIdCounter
from .base import (
    EvaluationFunctionCallback,
    EvaluationFunctionContext,
    EvaluationFunctionResult,
    Evaluator,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

_logger = get_logger(__name__)


class ParallelEvaluator(Evaluator):
    """An evaluator that dispatches tasks to an executor via asyncio.

    Submits the rows of the evaluation batch as tasks to the executor's task
    queue and collects results from a results queue. By default each row is
    submitted as its own task; the `bundle_size` constructor argument can be
    used to group several active evaluations into a single task that the worker
    executes sequentially.

    See [Parallel Evaluation](../workflows/parallel.md#parallelevaluator) for
    details on how this integrates with the asyncio event loop.
    """

    def __init__(
        self,
        *,
        function: EvaluationFunctionCallback,
        executor: Executor,
        bundle_size: int = 1,
        batch_id_callback: Callable[[], int] | None = None,
    ) -> None:
        """Initialize the ParallelEvaluator.

        With `bundle_size=1` (the default) every active evaluation is sent as
        its own executor task. Setting `bundle_size` to an integer `> 1` groups
        up to that many active evaluations into one task that the worker runs
        sequentially; `0` packs all active evaluations of a batch into a single
        task.

        Args:
            function:          The function used for objectives and constraints.
            executor:          The executor to dispatch tasks to.
            bundle_size:       Number of active evaluations per executor task.
            batch_id_callback: Callable that returns the next batch ID each time it is called.

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

    def eval(
        self, variables: NDArray[np.float64], evaluator_context: EvaluationBatchContext
    ) -> EvaluationBatchResult:
        """Evaluate all objective and constraints.

        An infrastructure failure is recorded as a failed realization (NaN); a
        user-code exception is re-raised, leaving the executor running. Raises
        [`ExecutorStopped`][ropt.exceptions.ExecutorStopped] if the executor
        cannot run the evaluation. See
        [error handling](../workflows/parallel.md#error-handling) for the full
        contract.

        Args:
            variables:      The matrix of variables to evaluate.
            evaluator_context: The evaluation context.

        Returns:
            The result of calling the wrapped evaluator function.

        Raises:
            WorkflowError: If called on the executor's own event loop thread.
        """
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
        metadata: dict[str, NDArray[Any]] = {}

        bundles = self._make_bundles(variables, evaluator_context, batch_id)
        _logger.debug("Dispatching %d work item(s) to executor", len(bundles))
        submission = Submission(
            [
                WorkItem(function=_run_bundle, args=(self._function, bundle))
                for bundle in bundles
            ]
        )
        self._executor.submit(submission)
        submission.collect(
            partial(
                _handle_result,
                results=results,
                metadata=metadata,
                objective_count=no,
                eval_count=variables.shape[0],
            ),
        )

        return EvaluationBatchResult(
            batch_id=batch_id,
            objectives=results[:, :no],
            constraints=results[:, no:] if nc > 0 else None,
            metadata=metadata,
        )

    def _make_bundles(
        self,
        variables: NDArray[np.float64],
        context: EvaluationBatchContext,
        batch_id: int,
    ) -> list[list[tuple[NDArray[np.float64], EvaluationFunctionContext]]]:
        bundles: list[list[tuple[NDArray[np.float64], EvaluationFunctionContext]]] = []
        bundle: list[tuple[NDArray[np.float64], EvaluationFunctionContext]] = []
        for eval_idx, function_context in _active_evaluations(context, batch_id):
            bundle.append((variables[eval_idx, :], function_context))
            if self._bundle_size and len(bundle) >= self._bundle_size:
                bundles.append(bundle)
                bundle = []
        if bundle:
            bundles.append(bundle)
        return bundles


def _run_bundle(
    function: EvaluationFunctionCallback,
    bundle: list[tuple[NDArray[np.float64], EvaluationFunctionContext]],
) -> list[EvaluationFunctionResult]:
    return list(starmap(function, bundle))


def _handle_result(
    work_item: WorkItem,
    results: NDArray[np.float64],
    metadata: dict[str, NDArray[Any]],
    objective_count: int,
    eval_count: int,
) -> None:
    bundle: list[tuple[NDArray[np.float64], EvaluationFunctionContext]] = (
        work_item.args[1]
    )
    if isinstance(work_item.result, ExecutorFailure):
        for _, function_context in bundle:
            results[function_context.eval_idx, :] = np.nan
        return
    if not isinstance(work_item.result, list) or len(work_item.result) != len(bundle):
        msg = (
            f"The evaluation function must return a list of {len(bundle)} "
            f"EvaluationFunctionResult objects."
        )
        raise WorkflowError(msg)
    for (_, function_context), result in zip(bundle, work_item.result, strict=True):
        if not isinstance(result, EvaluationFunctionResult):
            msg = (
                "The evaluation function must return EvaluationFunctionResult "
                f"objects, got {type(result).__name__}."
            )
            raise WorkflowError(msg)
        _scatter_result(
            function_context.eval_idx,
            result,
            results,
            metadata,
            objective_count,
            eval_count,
        )
