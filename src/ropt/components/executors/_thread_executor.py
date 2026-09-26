"""This module implements the thread-based executor."""

from __future__ import annotations

import queue
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from functools import partial
from typing import TYPE_CHECKING, Any

from ropt._logging import get_logger

from .base import ExecutorBase, WorkItem, _calls, _run_bundle, _stopped

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_logger = get_logger(__name__)


class ThreadExecutor(ExecutorBase):
    """An executor that dispatches work items to worker threads.

    The pool's own queue provides the worker cap and the arrival order across
    concurrent callers.

    Warning:
        Closing cannot interrupt a thread. Work that has already started runs to
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
        # Bundles running in a worker thread right now, over the whole executor.
        self._in_flight = 0
        _logger.debug("Started thread executor with %d worker(s)", workers)

    def _on_close(self) -> None:
        if self._in_flight > 0:
            # A thread cannot be cancelled, and the pool joins its threads when
            # the interpreter exits, so these work items decide when the program
            # is allowed to leave. Said out loud, because otherwise it is
            # indistinguishable from a hang.
            _logger.warning(
                "Closing with %d evaluation(s) still running: a thread cannot "
                "be interrupted, so they run to completion first.",
                self._in_flight,
            )

    def _release(self) -> None:
        _shutdown_pool(self._pool)

    def _run_bundles(
        self,
        bundles: list[list[WorkItem]],
        store: Callable[[int, Any], None],
    ) -> None:
        # Finished bundles arrive here rather than through
        # `concurrent.futures.wait`, which would block forever on a future
        # cancelled before a worker picked it up; a done callback still fires.
        done: queue.SimpleQueue[Future[list[Any]]] = queue.SimpleQueue()
        futures: dict[Future[list[Any]], int] = {}
        try:
            for index, bundle in enumerate(bundles):
                future = self._submit(bundle)
                futures[future] = index
                future.add_done_callback(done.put)
            for _ in range(len(bundles)):
                future = done.get()
                store(futures.pop(future), _bundle_result(future))
        finally:
            for future in futures:
                future.cancel()

    def _submit(self, bundle: list[WorkItem]) -> Future[list[Any]]:
        with self._lock:
            if self._closed:
                raise _stopped()
            try:
                return self._pool.submit(partial(self._run_tracked, _calls(bundle)))
            except RuntimeError:
                raise _stopped() from None

    def _run_tracked(
        self,
        calls: Sequence[tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]],
    ) -> list[Any]:
        with self._lock:
            self._in_flight += 1
        # Runs on the pool thread, so it marks that thread, not the submitter.
        self._thread_state.running_work_item = True
        try:
            return _run_bundle(calls)
        finally:
            self._thread_state.running_work_item = False
            with self._lock:
                self._in_flight -= 1


def _shutdown_pool(pool: ThreadPoolExecutor) -> None:
    pool.shutdown(wait=False, cancel_futures=True)


def _bundle_result(future: Future[list[Any]]) -> list[Any]:
    try:
        return future.result()
    except CancelledError:
        raise _stopped() from None
