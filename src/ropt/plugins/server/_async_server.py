"""This module implements the default asynchronous evaluator server."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, TypeVar

from ropt.enums import ExitCode
from ropt.exceptions import ComputeStepAborted
from ropt.plugins.server.base import Server, ServerBase, Task

if TYPE_CHECKING:
    import queue

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
        self._wait_event = asyncio.Event()
        self._wait_task: asyncio.Task[None] | None = None

    def start(self, task_group: asyncio.TaskGroup) -> None:
        """Start the server.

        Args:
            task_group: The task group to use.
        """
        super().start(task_group)
        workers = [_Worker(self._task_queue, server=self) for _ in range(self._workers)]
        self._worker_tasks = [
            task_group.create_task(worker.run()) for worker in workers
        ]
        self._wait_task = task_group.create_task(self._wait())

    async def _wait(self) -> None:
        try:
            await self._wait_event.wait()
            self._wait_event.clear()
        except asyncio.CancelledError:
            pass
        super().cancel()
        for worker_task in self._worker_tasks:
            if not worker_task.done():
                worker_task.cancel()
        self._worker_tasks = []
        queues: set[queue.Queue[TR]] = set()
        while not self._task_queue.empty():
            try:
                task = self._task_queue.get_nowait()
                if task.result_queue not in queues:
                    queues.add(task.result_queue)
                    task.put_exception(ComputeStepAborted(ExitCode.ABORT_FROM_ERROR))
                self._task_queue.task_done()
            except asyncio.QueueEmpty:
                break

    def cancel(self) -> None:
        """Stop the server."""
        self._wait_event.set()


class _Worker:
    def __init__(self, task_queue: asyncio.Queue[Task[R, TR]], server: Server) -> None:
        self._task_queue = task_queue
        self._server = server

    async def run(self) -> None:
        while self._server.is_running():
            task = await self._task_queue.get()
            try:
                result = task.function()
                task.put_result(result)
            except Exception as exc:  # noqa: BLE001
                task.put_exception(exc)
                self._server.cancel()
                self._task_queue.task_done()
