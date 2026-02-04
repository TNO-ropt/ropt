"""This module implements the default function evaluator."""

from __future__ import annotations

import queue
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ropt.enums import ExitCode
from ropt.evaluator import EvaluatorContext, EvaluatorResult
from ropt.exceptions import ComputeStepAborted
from ropt.plugins.server.base import ServerBase, Task, TaskResult

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
        function: Callable[[NDArray[np.float64], int, int], float],
        server: ServerBase[Task[float, _TaskResult]],
        maxsize: int = 0,
    ) -> None:
        """Initialize the DefaultFunctionEvaluator.

        Args:
            function: The function used for objectives and constraints.
            server:   Optional evaluator server to use.
            maxsize:  Maximum size of the result queue.
        """
        super().__init__()
        self._function = function
        self._server = server
        self._maxsize = maxsize

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

        no = context.config.objectives.weights.size
        nc = (
            0
            if context.config.nonlinear_constraints is None
            else context.config.nonlinear_constraints.lower_bounds.size
        )

        results_queue: queue.Queue[_TaskResult] = queue.Queue(self._maxsize)
        if self._server.loop is not None and self._server.task_group is not None:
            self._server.loop.call_soon_threadsafe(
                self._server.task_group.create_task,
                self._put_tasks(variables, context, results_queue),
            )

        results = np.zeros((variables.shape[0], no + nc), dtype=np.float64)
        for eval_idx in range(variables.shape[0]):
            if context.active is None or context.active[eval_idx]:
                while self._server.is_running():
                    try:
                        result = results_queue.get(timeout=1)
                        assert isinstance(result, _TaskResult)
                        if result.exception is not None:
                            raise result.exception
                        results[result.eval_idx] = result.value
                        break
                    except queue.Empty:
                        continue
            if not self._server.is_running():
                raise ComputeStepAborted(ExitCode.ABORT_FROM_ERROR)

        return EvaluatorResult(
            objectives=results[:, :no],
            constraints=results[:, no:] if nc > 0 else None,
        )

    async def _put_tasks(
        self,
        variables: NDArray[np.float64],
        context: EvaluatorContext,
        results_queue: queue.Queue[_TaskResult],
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
                    task = _Task(
                        result_queue=results_queue,
                        function=self._function,
                        args=(variables[eval_idx, :], int(realization), perturbation),
                        eval_idx=eval_idx,
                    )
                    await self._server.task_queue.put(task)
        except Exception as exc:  # noqa: BLE001
            results_queue.put(_TaskResult(value=0.0, eval_idx=eval_idx, exception=exc))

        if not self._server.is_running():
            raise ComputeStepAborted(ExitCode.ABORT_FROM_ERROR)


@dataclass(frozen=True, kw_only=True)
class _TaskResult(TaskResult):
    value: float = 0.0
    eval_idx: int
    exception: Exception | None = None


@dataclass(frozen=True, kw_only=True)
class _Task(Task[float, _TaskResult]):
    eval_idx: int

    def put_result(self, result: float) -> None:
        self.result_queue.put(_TaskResult(value=result, eval_idx=self.eval_idx))

    def put_exception(self, exc: Exception) -> None:
        self.result_queue.put(_TaskResult(eval_idx=self.eval_idx, exception=exc))
