"""This module implements the default multiprocessing executor."""

from __future__ import annotations

import asyncio
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import TYPE_CHECKING, Any

from ropt._logging import get_logger
from ropt.exceptions import ExecutorFailure

from ._worker import _mark_worker_process
from .base import Executor, ExecutorBase, Task

if TYPE_CHECKING:
    from collections.abc import Callable

_logger = get_logger(__name__)


class MultiprocessingExecutor(ExecutorBase):
    """An executor that employs a pool of multiprocessing workers.

    See [Parallel Evaluation](../usage/parallel.md#multiprocessingexecutor) for
    details, including the `if __name__ == "__main__":` guard that the entry
    point must use.
    """

    def __init__(
        self,
        *,
        workers: int = 1,
        queue_size: int = 0,
        max_tasks_per_child: int | None = None,
    ) -> None:
        """Initialize the executor.

        Args:
            workers:             Number of worker processes.
            queue_size:          Maximum task queue size (0 = unlimited).
            max_tasks_per_child: Restart workers after this many tasks (`None` = never).
        """
        super().__init__(queue_size=queue_size)
        self._workers = workers
        self._max_tasks_per_child = max_tasks_per_child
        self._worker_tasks: list[asyncio.Task[None]] = []
        self._executor: ProcessPoolExecutor | None = None

    async def start(self, task_group: asyncio.TaskGroup) -> None:
        """Start the executor.

        Args:
            task_group:          The task group to use.
        """
        self._executor = ProcessPoolExecutor(
            max_workers=self._workers,
            mp_context=multiprocessing.get_context("spawn"),
            max_tasks_per_child=self._max_tasks_per_child,
            initializer=_mark_worker_process,
        )
        _logger.debug(
            "Starting multiprocessing executor with %d worker(s)", self._workers
        )
        await self._check_worker_startup()
        workers = [
            _Worker(self._task_queue, self, self._executor)
            for _ in range(self._workers)
        ]
        self._worker_tasks = [
            task_group.create_task(worker.run()) for worker in workers
        ]
        await self._finish_start(task_group)

    async def _check_worker_startup(self) -> None:
        assert self._executor is not None
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(self._executor, _canary)
        except BrokenProcessPool as exc:
            self._executor.shutdown(wait=False)
            self._executor = None
            msg = (
                "MultiprocessingExecutor could not start its worker processes. "
                "Usually this is caused by a missing 'if __name__ == \"__main__\":' "
                "guard. See the parallel evaluation documentation for other causes."
            )
            raise RuntimeError(msg) from exc

    def cleanup(self) -> None:
        """Clean up the executor."""
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
        task_queue: asyncio.Queue[Task],
        parent: Executor,
        executor: ProcessPoolExecutor,
    ) -> None:
        self._task_queue = task_queue
        self._parent = parent
        self._executor = executor

    async def run(self) -> None:
        while True:
            task = await self._task_queue.get()
            try:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    self._executor, _run_function, task.function, task.args, task.kwargs
                )
                await asyncio.to_thread(task.put_result, result)
            except BrokenProcessPool:
                _logger.warning("Worker process pool broken; task result lost")
                await asyncio.to_thread(
                    task.put_result, ExecutorFailure("Background process was killed")
                )
            except Exception as exc:
                task.put_error(exc)
                raise
            finally:
                self._task_queue.task_done()


def _run_function(
    function: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:  # ruff: ignore[any-type]
    return function(*args, **kwargs)


def _canary() -> None:
    pass
