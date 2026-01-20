"""This module implements the default asynchronous evaluator server."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from ropt.plugins.server.base import EvaluatorServer, EvaluatorTask, EvaluatorTaskResult

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    import numpy as np
    from numpy.typing import NDArray


class DefaultFunctionEvaluatorServer(EvaluatorServer):
    """An evaluator server that calls an awaitable function."""

    def __init__(
        self,
        *,
        function: Callable[[NDArray[np.float64], int], Awaitable[float]] | None = None,
        workers: int = 1,
    ) -> None:
        """Initialize the DefaultAsyncEvaluator.

        Args:
            function: The function used for objectives and constraints.
            workers:  The number of workers to use.
        """
        super().__init__()
        self._function = function
        self._workers = workers
        self._worker_tasks: list[asyncio.Task[None]] = []
        self._running: bool = False
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def loop(self) -> asyncio.AbstractEventLoop | None:
        """Get the asyncio loop used by this server."""
        return self._loop

    def start(self, task_group: asyncio.TaskGroup) -> None:
        """Start the evaluation server.

        Args:
            task_group: The task group to use.

        Raises:
            RuntimeError: If the evaluator is already running or using an
                          external queue.
        """
        if self._running:
            msg = "Evaluator is already running."
            raise RuntimeError(msg)
        self._loop = asyncio.get_running_loop()
        workers = [_Worker(self._task_queue) for _ in range(self._workers)]
        self._worker_tasks = [
            task_group.create_task(worker.run()) for worker in workers
        ]
        self._running = True

    def stop(self) -> None:
        """Stop the evaluation server."""
        for worker_task in self._worker_tasks:
            worker_task.cancel()
        self._worker_tasks = []
        self._running = False


class _Worker:
    """A worker that executes asynchronous evaluation tasks."""

    def __init__(self, task_queue: asyncio.Queue[EvaluatorTask]) -> None:
        """Initialize the AsyncWorker.

        Args:
            task_queue: The queue to get tasks from.
        """
        self._task_queue = task_queue

    async def run(self) -> None:
        """Run the worker loop."""
        while True:
            task = await self._task_queue.get()
            try:
                result = await task.run()
                await asyncio.sleep(0.0)
                task.result_queue.put(
                    EvaluatorTaskResult(task_id=task.task_id, value=result)
                )
            finally:
                self._task_queue.task_done()
