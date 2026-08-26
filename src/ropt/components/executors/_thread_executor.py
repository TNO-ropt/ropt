"""This module implements the thread-based executor."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

from ropt._logging import get_logger

from .base import ExecutorBase, WorkItem

_logger = get_logger(__name__)


class ThreadExecutor(ExecutorBase):
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
        # Ids of the pool threads. A thread only ever asks about itself, and
        # registered its own id before it could run anything, so it always sees
        # at least that id without locking. This marks a thread for its whole
        # life rather than for one work item: wider than the question asked, and
        # harmless because the pool runs nothing else.
        self._worker_ids: frozenset[int] = frozenset()
        self._worker_ids_lock = threading.Lock()
        # Work items handed to a pool thread and not yet returned. Only the loop
        # thread reads or writes it, from `_run_worker` and `_cleanup`, so it
        # needs no lock.
        self._in_flight = 0

    async def start(self, task_group: asyncio.TaskGroup) -> None:
        """Start the executor.

        Args:
            task_group: The task group to use.
        """
        self._begin_start()
        pool = ThreadPoolExecutor(
            max_workers=self._workers, initializer=self._register_worker
        )
        self._pool = pool
        _logger.debug("Starting thread executor with %d worker(s)", self._workers)
        # One task per pool thread: each pulls work items and awaits its own,
        # so the number of tasks is what limits how many run at once.
        self._worker_tasks = [
            task_group.create_task(self._run_worker(pool)) for _ in range(self._workers)
        ]
        await self._finish_start(task_group)

    def on_worker_thread(self) -> bool:
        """Report whether the caller is running as one of this executor's workers.

        A thread started by a work item is not a worker: it holds none of this
        executor's workers, and is not recognized here.

        Returns:
            `True` if the calling thread is one of this executor's workers.
        """
        return threading.get_ident() in self._worker_ids

    def _register_worker(self) -> None:
        # Runs once per pool thread, as it is spawned.
        with self._worker_ids_lock:
            self._worker_ids |= {threading.get_ident()}

    def _cleanup(self) -> None:
        """Clean up the executor."""
        if self._in_flight > 0:
            # A thread cannot be cancelled, and the pool joins its threads when
            # the interpreter exits, so these work items decide when the program
            # is allowed to leave. Said out loud, because otherwise it is
            # indistinguishable from a hang.
            _logger.warning(
                "Stopping with %d evaluation(s) still running: a thread cannot "
                "be interrupted, so they run to completion first.",
                self._in_flight,
            )
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
        # The ids belong to that pool's threads: a restarted executor builds a
        # new pool, and the system may reuse the ids of the old one.
        with self._worker_ids_lock:
            self._worker_ids = frozenset()
        for worker_task in self._worker_tasks:
            if not worker_task.done():
                worker_task.cancel()
        self._worker_tasks = []
        self._cleanup_submissions()

    async def _run_in_pool(self, pool: ThreadPoolExecutor, work_item: WorkItem) -> Any:  # ruff: ignore[any-type]
        loop = asyncio.get_running_loop()
        self._in_flight += 1
        try:
            return await loop.run_in_executor(
                pool,
                partial(work_item.function, *work_item.args, **work_item.kwargs),
            )
        finally:
            self._in_flight -= 1

    async def _run_worker(self, pool: ThreadPoolExecutor) -> None:
        while True:
            submission, work_item = await self._work_queue.get()
            if submission.is_finished:
                # Its caller has already left, so running this wastes a worker.
                continue
            try:
                result = await self._run_in_pool(pool, work_item)
                self._deliver(submission, work_item, result)
            except asyncio.CancelledError:
                self._abort(submission)
                raise
            except BaseException as exc:
                # The caller is told either way, but an exception that is not an
                # `Exception` (SystemExit, KeyboardInterrupt) is not this run's
                # to swallow: it keeps unwinding into the task group.
                self._fail(submission, exc)
                if not isinstance(exc, Exception):
                    raise
