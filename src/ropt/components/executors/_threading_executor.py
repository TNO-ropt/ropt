"""This module implements the thread-based executor."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from ropt._logging import get_logger

from .base import ExecutorBase

_logger = get_logger(__name__)


class ThreadingExecutor(ExecutorBase):
    """An executor that dispatches work items to worker threads."""

    def __init__(self, *, workers: int = 1) -> None:
        """Initialize the executor.

        Args:
            workers: The number of workers to use.
        """
        super().__init__()
        self._workers = workers
        self._worker_tasks: list[asyncio.Task[None]] = []
        self._pool: ThreadPoolExecutor | None = None

    async def start(self, task_group: asyncio.TaskGroup) -> None:
        """Start the executor.

        Args:
            task_group: The task group to use.
        """
        self._begin_start()
        pool = ThreadPoolExecutor(max_workers=self._workers)
        self._pool = pool
        _logger.debug("Starting threading executor with %d worker(s)", self._workers)
        self._worker_tasks = [
            task_group.create_task(self._run_worker(pool)) for _ in range(self._workers)
        ]
        await self._finish_start(task_group)

    def _cleanup(self) -> None:
        """Clean up the executor."""
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
        for worker_task in self._worker_tasks:
            if not worker_task.done():
                worker_task.cancel()
        self._worker_tasks = []
        self._cleanup_submissions()

    async def _run_worker(self, pool: ThreadPoolExecutor) -> None:
        loop = asyncio.get_running_loop()
        while True:
            submission, work_item = await self._work_queue.get()
            if submission.is_finished:
                # Its caller has already left, so running this wastes a worker.
                continue
            try:
                result = await loop.run_in_executor(
                    pool,
                    partial(work_item.function, *work_item.args, **work_item.kwargs),
                )
                self._deliver(submission, work_item, result)
            except asyncio.CancelledError:
                self._abort(submission)
                raise
            except BaseException as exc:
                self._fail(submission, exc)
                if not isinstance(exc, Exception):
                    raise
