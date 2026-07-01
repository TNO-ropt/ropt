"""This module implements the default function evaluator."""

from __future__ import annotations

import queue
import threading
from itertools import starmap
from typing import TYPE_CHECKING, Any

import numpy as np

from ropt.enums import ExitCode
from ropt.evaluation import EvaluationBatchContext, EvaluationBatchResult
from ropt.exceptions import Abort, ServerFailure
from ropt.exit_info import ExitInfo
from ropt.workflow.servers import ResultsQueue, Server, Task

from .base import (
    EvaluationFunctionCallback,
    EvaluationFunctionContext,
    EvaluationFunctionResult,
    Evaluator,
    NameCallback,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class AsyncEvaluator(Evaluator):
    """An evaluator that dispatches tasks to a server via asyncio.

    Submits the rows of the evaluation batch as tasks to the server's task
    queue and collects results from a results queue. By default each row is
    submitted as its own task; the `bundle_size` constructor argument can be
    used to group several active evaluations into a single task that the
    worker executes sequentially.

    See [Parallel Evaluation](../usage/parallel.md#asyncevaluator) for
    details on how this integrates with the asyncio event loop.
    """

    # NOTE: A single instance of this class might be used from different
    # threads, if it is shared by different optimizers running in different
    # threads. The server is expected to be thread-safe, and the results queue
    # is thread-safe, so this should be safe. The batch ID is protected by a
    # lock.

    def __init__(
        self,
        *,
        function: EvaluationFunctionCallback,
        server: Server,
        bundle_size: int = 1,
        queue_size: int = 0,
        get_name: NameCallback | None = None,
    ) -> None:
        """Initialize the FunctionEvaluator.

        With `bundle_size=1` (the default) every active evaluation is sent
        as its own server task. Setting `bundle_size` to an integer `> 1`
        groups up to that many active evaluations into one task that the
        worker runs sequentially; `0` packs all active evaluations of a
        batch into a single task.

        The `get_name` callback receives the
        `EvaluationFunctionContext` objects for every evaluation in a task
        (a one-element sequence when `bundle_size=1`) and must return a
        single task name.

        Args:
            function:    The function used for objectives and constraints.
            server:      Optional evaluator server to use.
            bundle_size: Number of active evaluations per server task.
            queue_size:  Maximum size of the result queue.
            get_name:    Optional callable to generate names for tasks.

        Raises:
            ValueError: If `bundle_size` is negative.
        """
        super().__init__()
        if bundle_size < 0:
            msg = f"bundle_size must be >= 0, got {bundle_size}"
            raise ValueError(msg)
        self._function = function
        self._server = server
        self._bundle_size = bundle_size
        self._queue_size = queue_size
        self._batch_id = 0
        self._batch_lock = threading.Lock()
        self._get_name = get_name

    def __getstate__(self) -> dict[str, Any]:
        # threading.Lock is not picklable; drop it and recreate in __setstate__.
        state = self.__dict__.copy()
        state.pop("_batch_lock", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._batch_lock = threading.Lock()

    def eval(
        self, variables: NDArray[np.float64], evaluator_context: EvaluationBatchContext
    ) -> EvaluationBatchResult:
        """Evaluate all objective and constraints.

        Args:
            variables:      The matrix of variables to evaluate.
            evaluator_context: The evaluation context.

        Returns:
            The result of calling the wrapped evaluator function.

        Raises:
            Abort: raise if the server is not running.
        """
        if not self._server.is_running():
            raise Abort(
                ExitInfo(
                    exit_code=ExitCode.ABORT_FROM_ERROR,
                    message="Evaluation server is not running",
                )
            )

        with self._batch_lock:
            batch_id = self._batch_id
            self._batch_id += 1

        no = evaluator_context.context.objectives.weights.size
        nc = (
            0
            if evaluator_context.context.nonlinear_constraints is None
            else evaluator_context.context.nonlinear_constraints.lower_bounds.size
        )

        results_queue = ResultsQueue(self._queue_size)
        if self._server.loop is not None and self._server.task_group is not None:
            self._server.loop.call_soon_threadsafe(
                self._server.task_group.create_task,
                self._put_tasks(variables, evaluator_context, results_queue, batch_id),
            )

        results = np.zeros((variables.shape[0], no + nc), dtype=np.float64)
        metadata: dict[str, NDArray[Any]] = {}

        active_count = (
            variables.shape[0]
            if evaluator_context.active is None
            else int(evaluator_context.active.sum())
        )
        received = 0
        while received < active_count:
            while self._server.is_running():
                try:
                    if (task := results_queue.get(timeout=1)) is None:
                        raise Abort(
                            ExitInfo(
                                exit_code=ExitCode.ABORT_FROM_ERROR,
                                message="Evaluation server is not running",
                            )
                        )
                    received += _handle_result(
                        task, results, metadata, no, variables.shape[0]
                    )
                    break
                except queue.Empty:
                    continue
            if not self._server.is_running():
                raise Abort(
                    ExitInfo(
                        exit_code=ExitCode.ABORT_FROM_ERROR,
                        message="Evaluation server is not running",
                    )
                )

        return EvaluationBatchResult(
            batch_id=batch_id,
            objectives=results[:, :no],
            constraints=results[:, no:] if nc > 0 else None,
            metadata=metadata,
        )

    async def _put_tasks(
        self,
        variables: NDArray[np.float64],
        context: EvaluationBatchContext,
        results_queue: ResultsQueue,
        batch_id: int,
    ) -> None:
        try:
            await self._submit_bundles(variables, context, results_queue, batch_id)
        except Exception:
            results_queue.put(None)
            results_queue.close()
            raise

        if not self._server.is_running():
            raise Abort(
                ExitInfo(
                    exit_code=ExitCode.ABORT_FROM_ERROR,
                    message="Evaluation server is not running",
                )
            )

    async def _submit_bundles(
        self,
        variables: NDArray[np.float64],
        context: EvaluationBatchContext,
        results_queue: ResultsQueue,
        batch_id: int,
    ) -> None:
        bundle: list[tuple[NDArray[np.float64], EvaluationFunctionContext]] = []
        for eval_idx, realization in enumerate(context.realizations):
            if not self._server.is_running():
                break
            if context.active is not None and not context.active[eval_idx]:
                continue
            perturbation = (
                -1
                if context.perturbations is None
                else int(context.perturbations[eval_idx])
            )
            function_context = EvaluationFunctionContext(
                realization=int(realization),
                perturbation=perturbation,
                batch_id=batch_id,
                eval_idx=eval_idx,
            )
            bundle.append((variables[eval_idx, :], function_context))
            if self._bundle_size and len(bundle) >= self._bundle_size:
                await self._server.task_queue.put(
                    self._make_task(bundle, results_queue)
                )
                bundle = []
        if bundle:
            await self._server.task_queue.put(self._make_task(bundle, results_queue))

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
) -> int:
    bundle: list[tuple[NDArray[np.float64], EvaluationFunctionContext]] = task.args[1]
    if isinstance(task.result, ServerFailure):
        for _, function_context in bundle:
            results[function_context.eval_idx, :] = np.nan
        return len(bundle)
    assert isinstance(task.result, list)
    assert len(task.result) == len(bundle)
    for (_, function_context), result in zip(bundle, task.result, strict=True):
        eval_idx = function_context.eval_idx
        assert isinstance(result, EvaluationFunctionResult)
        results[eval_idx, :objective_count] = result.objectives
        if result.constraints is not None:
            results[eval_idx, objective_count:] = result.constraints
        if result.metadata is not None:
            for key, value in result.metadata.items():
                if key not in metadata:
                    metadata[key] = np.zeros(
                        eval_count,
                        dtype=(
                            np.array(value).dtype
                            if isinstance(value, (int, float, complex, np.number))
                            else object
                        ),
                    )
                metadata[key][eval_idx] = value
    return len(bundle)
