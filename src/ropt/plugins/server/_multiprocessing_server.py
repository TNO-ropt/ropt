"""This module implements the default multiprocessing evaluator server."""

from __future__ import annotations

import asyncio
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING, Any, TypeVar

from ropt.plugins.server.base import Server, ServerBase, Task

if TYPE_CHECKING:
    from collections.abc import Callable

R = TypeVar("R")
TR = TypeVar("TR")


class DefaultMultiprocessingServer(ServerBase[Task[R, TR]]):
    """An evaluator server that employs a pool of multiprocessing workers."""

    def __init__(self, *, workers: int = 1, queue_size: int = 0) -> None:
        """Initialize the server.

        Args:
            workers:    The number of workers to use.
            queue_size: Maximum size of the tasks queue.
        """
        super().__init__(queue_size=queue_size)
        self._workers = workers
        self._worker_tasks: list[asyncio.Task[None]] = []
        self._executor: ProcessPoolExecutor | None = None

    async def start(self, task_group: asyncio.TaskGroup) -> None:
        """Start the server.

        Args:
            task_group: The task group to use.
        """
        context = multiprocessing.get_context("spawn")
        self._executor = ProcessPoolExecutor(
            max_workers=self._workers, mp_context=context
        )
        workers = [
            _Worker(self._task_queue, self, self._executor)
            for _ in range(self._workers)
        ]
        self._worker_tasks = [
            task_group.create_task(worker.run()) for worker in workers
        ]
        await self._finish_start(task_group)

    def cleanup(self) -> None:
        """Cleanup the server."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        for worker_task in self._worker_tasks:
            if not worker_task.done():
                worker_task.cancel()
        self._worker_tasks = []
        self._drain_and_kill()


class _Worker:
    def __init__(
        self,
        task_queue: asyncio.Queue[Task[R, TR]],
        server: Server,
        executor: ProcessPoolExecutor,
    ) -> None:
        self._task_queue = task_queue
        self._server = server
        self._executor = executor

    async def run(self) -> None:
        while True:
            task = await self._task_queue.get()
            try:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    self._executor, _run_function, task.function, task.args, task.kwargs
                )
                task.put_result(result)
            except Exception:
                task.put_result(None)
                self._server.cancel()
                raise
            finally:
                self._task_queue.task_done()


def _run_function(
    function: Callable[..., R], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> R:
    return function(*args, **kwargs)
