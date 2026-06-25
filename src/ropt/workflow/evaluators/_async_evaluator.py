"""This module implements the default function evaluator."""

from __future__ import annotations

import queue
from typing import TYPE_CHECKING, Any

import numpy as np

from ropt.enums import ExitCode
from ropt.evaluation import EvaluationBatchContext, EvaluationBatchResult
from ropt.exceptions import Abort, ServerFailure
from ropt.workflow.servers import ResultsQueue, Server, Task

from .base import (
    Evaluator,
    EvaluatorFunctionCallback,
    EvaluatorFunctionContext,
    EvaluatorFunctionResult,
    NameCallback,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class AsyncEvaluator(Evaluator):
    """An evaluator that dispatches tasks to a server via asyncio.

    Submits each row of the evaluation batch as a separate task to the
    server's task queue and collects results from a results queue.

    See [Parallel Evaluation](../usage/parallel.md#asyncevaluator) for
    details on how this integrates with the asyncio event loop.
    """

    def __init__(
        self,
        *,
        function: EvaluatorFunctionCallback,
        server: Server,
        queue_size: int = 0,
        get_name: NameCallback | None = None,
    ) -> None:
        """Initialize the FunctionEvaluator.

        Args:
            function:   The function used for objectives and constraints.
            server:     Optional evaluator server to use.
            queue_size: Maximum size of the result queue.
            get_name:   Optional callable to generate names for tasks.
        """
        super().__init__()
        self._function = function
        self._server = server
        self._queue_size = queue_size
        self._batch_id = -1
        self._get_name = get_name

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
            raise Abort(ExitCode.ABORT_FROM_ERROR)

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
                self._put_tasks(variables, evaluator_context, results_queue),
            )

        results = np.zeros((variables.shape[0], no + nc), dtype=np.float64)
        evaluation_info: dict[str, NDArray[Any]] = {}

        for _ in range(
            variables.shape[0]
            if evaluator_context.active is None
            else evaluator_context.active.sum()
        ):
            while self._server.is_running():
                try:
                    if (task := results_queue.get(timeout=1)) is None:
                        raise Abort(ExitCode.ABORT_FROM_ERROR)
                    _handle_result(
                        task, results, evaluation_info, no, variables.shape[0]
                    )
                    break
                except queue.Empty:
                    continue
            if not self._server.is_running():
                raise Abort(ExitCode.ABORT_FROM_ERROR)

        return EvaluationBatchResult(
            batch_id=self._batch_id,
            objectives=results[:, :no],
            constraints=results[:, no:] if nc > 0 else None,
            evaluation_info=evaluation_info,
        )

    async def _put_tasks(
        self,
        variables: NDArray[np.float64],
        context: EvaluationBatchContext,
        results_queue: ResultsQueue,
    ) -> None:
        try:
            for eval_idx, realization in enumerate(context.realizations):
                if self._server.is_running() and (
                    context.active is None or context.active[eval_idx]
                ):
                    task = self._get_task(
                        variables, context, results_queue, eval_idx, int(realization)
                    )
                    await self._server.task_queue.put(task)
        except Exception:
            results_queue.put(None)
            results_queue.close()
            raise

        if not self._server.is_running():
            raise Abort(ExitCode.ABORT_FROM_ERROR)

    def _get_task(
        self,
        variables: NDArray[np.float64],
        context: EvaluationBatchContext,
        results_queue: ResultsQueue,
        eval_idx: int,
        realization: int,
    ) -> Task:
        perturbation = (
            -1
            if context.perturbations is None
            else int(context.perturbations[eval_idx])
        )
        task_name = (
            None
            if self._get_name is None
            else self._get_name(realization, perturbation, self._batch_id, eval_idx)
        )
        return Task(
            results_queue=results_queue,
            function=self._function,
            args=(
                variables[eval_idx, :],
                EvaluatorFunctionContext(
                    realization, perturbation, self._batch_id, eval_idx
                ),
            ),
            name=task_name,
        )


def _handle_result(
    task: Task,
    results: NDArray[np.float64],
    evaluation_info: dict[str, NDArray[Any]],
    objective_count: int,
    eval_count: int,
) -> None:
    eval_idx = task.args[1].eval_idx
    if isinstance(task.result, ServerFailure):
        results[eval_idx, :] = np.nan
    else:
        assert isinstance(task.result, EvaluatorFunctionResult)
        results[eval_idx, :objective_count] = task.result.objectives
        if task.result.constraints is not None:
            results[eval_idx, objective_count:] = task.result.constraints
        if task.result.evaluation_info is not None:
            for key, value in task.result.evaluation_info.items():
                if key not in evaluation_info:
                    evaluation_info[key] = np.zeros(
                        eval_count,
                        dtype=(
                            np.array(value).dtype
                            if isinstance(value, (int, float, complex, np.number))
                            else object
                        ),
                    )
                evaluation_info[key][eval_idx] = value
