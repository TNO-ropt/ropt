"""This module implements the thread-based executor."""

from __future__ import annotations

import queue
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from typing import TYPE_CHECKING, Any

from ropt._logging import get_logger

from .base import ExecutorBase, WorkItem, _calls, _run_bundle, _stopped

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ropt.components.concurrency import AbortSignal

_logger = get_logger(__name__)


class ThreadExecutor(ExecutorBase):
    """An executor that dispatches work items to worker threads.

    The pool's own queue provides the worker cap and the arrival order across
    concurrent callers.

    Warning:
        A thread cannot be interrupted. Work that has already started runs to
        completion, and the interpreter waits for it before exiting.
    """

    def __init__(self, *, workers: int = 1, bundle_size: int = 1) -> None:
        """Initialize the executor.

        Args:
            workers:     The number of worker threads.
            bundle_size: Calls per worker task, `0` for a whole batch.

        Raises:
            ValueError: If `workers` is less than one.
        """
        super().__init__(bundle_size=bundle_size)
        if workers < 1:
            msg = f"The number of workers must be at least one: {workers}"
            raise ValueError(msg)
        self._pool = ThreadPoolExecutor(max_workers=workers)
        _logger.debug("Started thread executor with %d worker(s)", workers)

    def _run_bundles(
        self,
        bundles: list[list[WorkItem]],
        store: Callable[[int, Any], None],
        abort_signal: AbortSignal | None,
    ) -> None:
        # Finished bundles arrive here rather than through
        # `concurrent.futures.wait`, which would block forever on a future
        # cancelled before a worker picked it up; a done callback still fires.
        done: queue.SimpleQueue[Future[list[Any]] | None] = queue.SimpleQueue()
        futures: dict[Future[list[Any]], int] = {}
        # The sentinel is what releases `done.get()` below; nothing else can.
        wake = partial(done.put, None)
        if abort_signal is not None:
            abort_signal.add_callback(wake)
        try:
            for index, bundle in enumerate(bundles):
                if abort_signal is not None and abort_signal.aborting:
                    break
                future = self._submit(bundle)
                futures[future] = index
                future.add_done_callback(done.put)
            while futures:
                item = done.get()
                if item is None:
                    # Dropping a bundle that is still queued is what a stop can
                    # do here. One already on a worker cannot be interrupted, so
                    # it is waited for and its outcome kept: discarding it would
                    # lose whatever it raised.
                    for future in list(futures):
                        future.cancel()
                    continue
                index = futures.pop(item)
                if not item.cancelled():
                    store(index, item.result())
        finally:
            if abort_signal is not None:
                abort_signal.remove_callback(wake)
            for future in futures:
                future.cancel()

    def _submit(self, bundle: list[WorkItem]) -> Future[list[Any]]:
        try:
            return self._pool.submit(partial(self._run_tracked, _calls(bundle)))
        except RuntimeError:
            # The pool is gone, which at interpreter shutdown is how a caller
            # that outlived its program is released rather than left waiting.
            raise _stopped() from None

    def _run_tracked(
        self,
        calls: Sequence[tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]],
    ) -> list[Any]:
        # Runs on the pool thread, so it marks that thread, not the submitter.
        self._thread_state.running_work_item = True
        try:
            return _run_bundle(calls)
        finally:
            self._thread_state.running_work_item = False
