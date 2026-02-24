"""This module implements the default function evaluator."""

from __future__ import annotations

import queue
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from ropt.enums import ExitCode
from ropt.evaluator import EvaluatorContext, EvaluatorResult
from ropt.exceptions import ComputeStepAborted
from ropt.plugins.server.base import ResultsQueue, ServerBase, Task

from .base import Evaluator

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


class DefaultAsyncEvaluator(Evaluator):
    """An evaluator that calls a function in a asyncio loop.

    This Evaluator stores a single awaitable function that returns a value for
    each objective and constraint.
    """

    def __init__(
        self,
        *,
        function: Callable[..., NDArray[np.float64] | dict[str, Any]],
        server: ServerBase,
        queue_size: int = 0,
        evaluation_info: dict[str, np.dtype] | None = None,
    ) -> None:
        """Initialize the DefaultFunctionEvaluator.

        Args:
            function:        The function used for objectives and constraints.
            server:          Optional evaluator server to use.
            queue_size:      Maximum size of the result queue.
            evaluation_info: Optional dictionary of evaluations info keys and data types.
        """
        super().__init__()
        self._function = function
        self._server = server
        self._queue_size = queue_size
        self._evaluation_info = {} if evaluation_info is None else evaluation_info
        self._batch_id = -1

    def eval(
        self, variables: NDArray[np.float64], context: EvaluatorContext
    ) -> EvaluatorResult:
        """Evaluate all objective and constraints.

        Args:
            variables: The matrix of variables to evaluate.
            context:   The evaluation context.

        Returns:
            The result of calling the wrapped evaluator function.

        Raises:
            ComputeStepAborted: raise if the server is not running.
        """
        if not self._server.is_running():
            raise ComputeStepAborted(ExitCode.ABORT_FROM_ERROR)

        self._batch_id += 1

        no = context.config.objectives.weights.size
        nc = (
            0
            if context.config.nonlinear_constraints is None
            else context.config.nonlinear_constraints.lower_bounds.size
        )

        results_queue = ResultsQueue(self._queue_size)
        if self._server.loop is not None and self._server.task_group is not None:
            self._server.loop.call_soon_threadsafe(
                self._server.task_group.create_task,
                self._put_tasks(variables, context, results_queue),
            )

        results = np.zeros((variables.shape[0], no + nc), dtype=np.float64)
        evaluation_info: dict[str, NDArray[Any]] = {
            key: np.zeros(variables.shape[0], dtype=dtype)
            for key, dtype in self._evaluation_info.items()
        }

        for _ in range(
            variables.shape[0] if context.active is None else context.active.sum()
        ):
            while self._server.is_running():
                try:
                    if (task := results_queue.get(timeout=1)) is None:
                        raise ComputeStepAborted(ExitCode.ABORT_FROM_ERROR)
                    _handle_result(task, results, evaluation_info)
                    break
                except queue.Empty:
                    continue
            if not self._server.is_running():
                raise ComputeStepAborted(ExitCode.ABORT_FROM_ERROR)

        return EvaluatorResult(
            batch_id=self._batch_id,
            objectives=results[:, :no],
            constraints=results[:, no:] if nc > 0 else None,
            evaluation_info=evaluation_info,
        )

    async def _put_tasks(
        self,
        variables: NDArray[np.float64],
        context: EvaluatorContext,
        results_queue: ResultsQueue,
    ) -> None:
        try:
            for eval_idx, realization in enumerate(context.realizations):
                if not self._server.is_running():
                    break
                perturbation = (
                    -1
                    if context.perturbations is None
                    else int(context.perturbations[eval_idx])
                )
                if context.active is None or context.active[eval_idx]:
                    task = Task(
                        results_queue=results_queue,
                        function=self._function,
                        args=(variables[eval_idx, :],),
                        kwargs={
                            "realization": int(realization),
                            "perturbation": perturbation,
                            "batch_id": self._batch_id,
                            "eval_idx": eval_idx,
                        },
                    )
                    await self._server.task_queue.put(task)
        except Exception:
            results_queue.put(None)
            results_queue.close()
            raise

        if not self._server.is_running():
            raise ComputeStepAborted(ExitCode.ABORT_FROM_ERROR)


def _handle_result(
    task: Task,
    results: NDArray[np.float64],
    evaluation_info: dict[str, NDArray[Any]],
) -> None:
    eval_idx = task.kwargs["eval_idx"]
    if isinstance(task.result, np.ndarray):
        results[eval_idx, :] = task.result
    else:
        assert isinstance(task.result, Mapping)
        for key, value in task.result.items():
            if key == "result":
                results[eval_idx, :] = value
            elif key in evaluation_info:
                evaluation_info[key][eval_idx] = value
