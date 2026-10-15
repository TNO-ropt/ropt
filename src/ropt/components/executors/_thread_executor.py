"""This module implements the thread-based executor."""

from __future__ import annotations

import queue
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from functools import partial
from typing import TYPE_CHECKING, Any, override

from ropt._logging import get_logger

from .base import (
    _NOT_RUN,
    ExecutorBase,
    WorkItem,
    WorkNotRun,
    _log_left_running,
    _run_item,
    _running_on,
    _stopped,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ropt.components.concurrency import AbortSignal

_logger = get_logger(__name__)


class ThreadExecutor(ExecutorBase):
    """An executor that dispatches work items to worker threads.

    The pool's own queue provides the worker cap and the arrival order across
    concurrent callers.

    Warning:
        A thread cannot be interrupted. Work that has already started runs to
        completion, and the interpreter waits for it before exiting. An abort
        does not wait for it: `run` returns at once, and an exception the work
        raises afterwards is logged.
    """

    def __init__(self, *, workers: int = 1) -> None:
        """Initialize the executor.

        Args:
            workers: The number of worker threads.

        Raises:
            ValueError: If `workers` is less than one.
        """
        super().__init__()
        if workers < 1:
            msg = f"The number of workers must be at least one: {workers}"
            raise ValueError(msg)
        self._pool = ThreadPoolExecutor(max_workers=workers)
        _logger.debug("Started thread executor with %d worker(s)", workers)

    @override
    def _run_items(
        self,
        items: list[WorkItem],
        store: Callable[[int, Any], None],
        abort_signal: AbortSignal | None,
    ) -> None:
        # Finished items arrive here rather than through
        # `concurrent.futures.wait`, which would block forever on a future
        # cancelled before a worker picked it up; a done callback still fires.
        done: queue.SimpleQueue[Future[Any] | None] = queue.SimpleQueue()
        futures: dict[Future[Any], int] = {}
        # The sentinel is what releases `done.get()` below; nothing else can.
        wake = partial(done.put, None)
        if abort_signal is not None:
            abort_signal.add_callback(wake)
        try:
            for index, work_item in enumerate(items):
                if abort_signal is not None and abort_signal.aborting:
                    break
                future = self._submit(work_item, abort_signal)
                futures[future] = index
                future.add_done_callback(done.put)
            while futures:
                item = done.get()
                if item is None:
                    break
                index = futures.pop(item)
                if not item.cancelled():
                    store(index, item.result())
        finally:
            if abort_signal is not None:
                abort_signal.remove_callback(wake)
            for future in futures:
                # An item already on a worker cannot be interrupted, so it is
                # left to finish.
                if not future.cancel():
                    future.add_done_callback(partial(_log_left_running, Future.result))

    def _submit(self, item: WorkItem, abort_signal: AbortSignal | None) -> Future[Any]:
        try:
            return self._pool.submit(partial(self._run_tracked, item, abort_signal))
        except RuntimeError:
            # The pool is gone, which at interpreter shutdown is how a caller
            # that outlived its program is released rather than left waiting.
            raise _stopped() from None

    def _run_tracked(self, item: WorkItem, abort_signal: AbortSignal | None) -> Any:  # ruff: ignore[any-type]
        # A worker can take the next task before `_run_items` cancels it, so
        # whether an item starts is decided here rather than there.
        if abort_signal is not None and abort_signal.aborting:
            return WorkNotRun(_NOT_RUN)
        # Runs on the pool thread, so it marks that thread's context, not the caller's.
        token = _running_on.set(self)
        try:
            with nullcontext() if abort_signal is None else abort_signal.as_parent():
                return _run_item(item.function, item.args, item.kwargs)
        finally:
            _running_on.reset(token)
