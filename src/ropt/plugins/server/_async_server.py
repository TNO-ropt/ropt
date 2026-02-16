"""This module implements the default asynchronous evaluator server."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from ropt.plugins.server.base import Server, ServerBase, Task

if TYPE_CHECKING:
    import asyncio

R = TypeVar("R")
TR = TypeVar("TR")


class DefaultAsyncServer(ServerBase[Task[R, TR]]):
    """An evaluator server that employs asynchronous workers."""

    def __init__(self, *, workers: int = 1, maxsize: int = 0) -> None:
        """Initialize the server.

        Args:
            workers:  The number of workers to use.
            maxsize:  Maximum size of the tasks queue.
        """
        super().__init__(maxsize=maxsize)
        self._workers = workers
        self._worker_tasks: list[asyncio.Task[None]] = []

    async def start(self, task_group: asyncio.TaskGroup) -> None:
        """Start the server.

        Args:
            task_group: The task group to use.
        """
        workers = [_Worker(self._task_queue, server=self) for _ in range(self._workers)]
        self._worker_tasks = [
            task_group.create_task(worker.run()) for worker in workers
        ]
        await self._finish_start(task_group)

    def cleanup(self) -> None:
        """Cleanup the server."""
        for worker_task in self._worker_tasks:
            if not worker_task.done():
                worker_task.cancel()
        self._worker_tasks = []
        self._drain_and_kill()


class _Worker:
    def __init__(self, task_queue: asyncio.Queue[Task[R, TR]], server: Server) -> None:
        self._task_queue = task_queue
        self._server = server

    async def run(self) -> None:
        while self._server.is_running():
            task = await self._task_queue.get()
            try:
                result = task.function(*task.args, **task.kwargs)
                task.put_result(result)
            except Exception:
                task.put_result(None)
                self._server.cancel()
                raise
            finally:
                self._task_queue.task_done()
