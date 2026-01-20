"""This module implements the default function evaluator."""

from __future__ import annotations

import asyncio
import queue
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from ropt.evaluator import EvaluatorContext, EvaluatorResult
from ropt.plugins.server.base import EvaluatorServer, EvaluatorTask, EvaluatorTaskResult

from .base import Evaluator

if TYPE_CHECKING:
    import uuid
    from collections.abc import Awaitable, Callable

    from numpy.typing import NDArray


class DefaultAsyncEvaluator(Evaluator):
    """An evaluator that calls a function in a asyncio loop.

    This Evaluator stores a single awaitable function that returns a value for
    each objective and constraint.
    """

    def __init__(
        self,
        *,
        function: Callable[[NDArray[np.float64], int], Awaitable[float]],
        server: EvaluatorServer,
    ) -> None:
        """Initialize the DefaultFunctionEvaluator.

        Args:
            function: The function used for objectives and constraints.
            server:   Optional evaluator server to use.
        """
        super().__init__()
        self._function = function
        self._server = server

    def eval(
        self, variables: NDArray[np.float64], context: EvaluatorContext
    ) -> EvaluatorResult:
        """Evaluate all objective and constraints.

        Args:
            variables: The matrix of variables to evaluate.
            context:   The evaluation context.

        Returns:
            The result of calling the wrapped evaluator function.
        """
        no = context.config.objectives.weights.size
        nc = (
            0
            if context.config.nonlinear_constraints is None
            else context.config.nonlinear_constraints.lower_bounds.size
        )
        results = np.zeros((variables.shape[0], no + nc), dtype=np.float64)

        results_queue: queue.Queue[EvaluatorTaskResult] = queue.Queue()
        indices: dict[uuid.UUID, int] = {}
        for eval_idx, realization in enumerate(context.realizations):
            if context.active is None or context.active[eval_idx]:
                function = partial(
                    self._function, variables[eval_idx, :], int(realization)
                )
                task = _Task(result_queue=results_queue, function=function)
                indices[task.task_id] = eval_idx
                loop = self._server.loop
                assert loop is not None
                loop.call_soon_threadsafe(self._server.task_queue.put_nowait, task)
        while indices:
            result = results_queue.get()
            results[indices.pop(result.task_id)] = result.value

        return EvaluatorResult(
            objectives=results[:, :no],
            constraints=results[:, no:] if nc > 0 else None,
        )


@dataclass(frozen=True, kw_only=True)
class _Task(EvaluatorTask):
    function: Callable[[], Awaitable[float]]

    async def run(self) -> float:
        await asyncio.sleep(0.0)
        return await self.function()
