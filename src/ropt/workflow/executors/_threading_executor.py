"""This module implements the thread-based executor."""

from __future__ import annotations

import asyncio

from ropt._logging import get_logger

from .base import Executor, ExecutorBase, Task

_logger = get_logger(__name__)


class ThreadingExecutor(ExecutorBase):
    """An executor that dispatches tasks to worker threads."""

    def __init__(self, *, workers: int = 1, queue_size: int = 0) -> None:
        """Initialize the executor.

        Args:
            workers:    The number of workers to use.
            queue_size: Maximum size of the tasks queue.
        """
        super().__init__(queue_size=queue_size)
        self._workers = workers
        self._worker_tasks: list[asyncio.Task[None]] = []

    async def start(self, task_group: asyncio.TaskGroup) -> None:
        """Start the executor.

        Args:
            task_group: The task group to use.
        """
        workers = [_Worker(self._task_queue, parent=self) for _ in range(self._workers)]
        _logger.debug("Starting threading executor with %d worker(s)", self._workers)
        self._worker_tasks = [
            task_group.create_task(worker.run()) for worker in workers
        ]
        await self._finish_start(task_group)

    def cleanup(self) -> None:
        """Clean up the executor."""
        for worker_task in self._worker_tasks:
            if not worker_task.done():
                worker_task.cancel()
        self._worker_tasks = []
        self._drain_and_kill()


class _Worker:
    def __init__(self, task_queue: asyncio.Queue[Task], parent: Executor) -> None:
        self._task_queue = task_queue
        self._parent = parent

    async def run(self) -> None:
        while self._parent.is_running():
            task = await self._task_queue.get()
            try:
                result = await asyncio.to_thread(
                    task.function, *task.args, **task.kwargs
                )
                await asyncio.to_thread(task.put_result, result)
            except Exception:
                task.cancel_all()
                self._parent.cancel()
                raise
            finally:
                self._task_queue.task_done()
