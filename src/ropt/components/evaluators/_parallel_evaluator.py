"""This module implements the default function evaluator."""

from __future__ import annotations

from functools import partial
from itertools import starmap
from typing import TYPE_CHECKING, Any

import numpy as np

from ropt._logging import get_logger
from ropt.components.executors import Executor, ResultsQueue, Task
from ropt.components.executors._collect import submit_and_collect
from ropt.evaluation import EvaluationBatchContext, EvaluationBatchResult
from ropt.exceptions import ExecutorFailure, ExecutorStopped

from ._common import _active_evaluations, _scatter_result
from ._counter import BatchIdCounter
from .base import (
    EvaluationFunctionCallback,
    EvaluationFunctionContext,
    EvaluationFunctionResult,
    Evaluator,
    NameCallback,
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

    def __init__(  # ruff: ignore[too-many-arguments]
        self,
        *,
        function: EvaluationFunctionCallback,
        executor: Executor,
        bundle_size: int = 1,
        queue_size: int = 0,
        get_name: NameCallback | None = None,
        batch_id_callback: Callable[[], int] | None = None,
    ) -> None:
        """Initialize the ParallelEvaluator.

        With `bundle_size=1` (the default) every active evaluation is sent as
        its own executor task. Setting `bundle_size` to an integer `> 1` groups
        up to that many active evaluations into one task that the worker runs
        sequentially; `0` packs all active evaluations of a batch into a single
        task.

        The `get_name` callback receives the `EvaluationFunctionContext` objects
        for every evaluation in a task (a one-element sequence when
        `bundle_size=1`) and must return a single task name. For the
        `HPCExecutor` the name is also the task id and filename base, so it must
        be unique within the executor.

        Args:
            function:          The function used for objectives and constraints.
            executor:          The executor to dispatch tasks to.
            bundle_size:       Number of active evaluations per executor task.
            queue_size:        Maximum size of the result queue.
            get_name:          Optional callable to generate names for tasks.
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
        self._queue_size = queue_size
        self._batch_id_callback = (
            batch_id_callback if batch_id_callback is not None else BatchIdCounter()
        )
        self._get_name = get_name

    def eval(
        self, variables: NDArray[np.float64], evaluator_context: EvaluationBatchContext
    ) -> EvaluationBatchResult:
        """Evaluate all objective and constraints.

        Results are collected following the two-class error contract described
        in [Parallel Evaluation](../workflows/parallel.md#error-handling). An
        infrastructure failure arrives as an
        [`ExecutorFailure`][ropt.exceptions.ExecutorFailure] result and is
        recorded as a failed realization (NaN), while a user-code exception
        arrives on the results queue and aborts the current evaluation by
        re-raising the original exception unchanged. The executor is left
        running, so a consumer may reuse it for further evaluations.

        Args:
            variables:      The matrix of variables to evaluate.
            evaluator_context: The evaluation context.

        Returns:
            The result of calling the wrapped evaluator function.

        Raises:
            ExecutorStopped: If the executor is not running and no task
                exception is available to re-raise.
        """
        if not self._executor.is_running():
            raise ExecutorStopped

        batch_id = self._batch_id_callback()

        no = evaluator_context.context.objectives.weights.size
        nc = (
            0
            if evaluator_context.context.nonlinear_constraints is None
            else evaluator_context.context.nonlinear_constraints.lower_bounds.size
        )

        results_queue = ResultsQueue(self._queue_size)
        results = np.zeros((variables.shape[0], no + nc), dtype=np.float64)
        metadata: dict[str, NDArray[Any]] = {}

        bundles = self._make_bundles(variables, evaluator_context, batch_id)
        _logger.debug("Dispatching %d task(s) to executor", len(bundles))
        submit_and_collect(
            self._executor,
            self._put_tasks(bundles, results_queue),
            results_queue,
            len(bundles),
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

    async def _put_tasks(
        self,
        bundles: list[list[tuple[NDArray[np.float64], EvaluationFunctionContext]]],
        results_queue: ResultsQueue,
    ) -> None:
        try:
            for bundle in bundles:
                if not self._executor.is_running():
                    break
                await self._executor.task_queue.put(
                    self._make_task(bundle, results_queue)
                )
        except Exception:
            results_queue.put(None)
            results_queue.close()
            raise

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

    def _make_task(
        self,
        bundle: list[tuple[NDArray[np.float64], EvaluationFunctionContext]],
        results_queue: ResultsQueue,
    ) -> Task:
        task_name = (
            None
            if self._get_name is None
            else self._get_name([function_context for _, function_context in bundle])
        )
        for _, function_context in bundle:
            function_context.name = task_name
        return Task(
            results_queue=results_queue,
            function=_run_bundle,
            args=(self._function, bundle),
            name=task_name,
        )


def _run_bundle(
    function: EvaluationFunctionCallback,
    bundle: list[tuple[NDArray[np.float64], EvaluationFunctionContext]],
) -> list[EvaluationFunctionResult]:
    return list(starmap(function, bundle))


def _handle_result(
    task: Task,
    results: NDArray[np.float64],
    metadata: dict[str, NDArray[Any]],
    objective_count: int,
    eval_count: int,
) -> None:
    bundle: list[tuple[NDArray[np.float64], EvaluationFunctionContext]] = task.args[1]
    if isinstance(task.result, ExecutorFailure):
        for _, function_context in bundle:
            results[function_context.eval_idx, :] = np.nan
        return
    assert isinstance(task.result, list)
    assert len(task.result) == len(bundle)
    for (_, function_context), result in zip(bundle, task.result, strict=True):
        assert isinstance(result, EvaluationFunctionResult)
        _scatter_result(
            function_context.eval_idx,
            result,
            results,
            metadata,
            objective_count,
            eval_count,
        )
