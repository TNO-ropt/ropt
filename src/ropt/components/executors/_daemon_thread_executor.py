"""This module implements an executor that starts threads for each batch."""

from __future__ import annotations

import contextvars
import queue
import threading
from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from functools import partial
from typing import TYPE_CHECKING, Any, override

from .base import ExecutorBase, WorkItem, _log_left_running, _run_item

if TYPE_CHECKING:
    from collections.abc import Callable

    from ropt.components.concurrency import AbortSignal

type _Job = tuple[Future[Any], Callable[[], Any]]


class DaemonThreadExecutor(ExecutorBase):
    """An executor that runs each batch on daemon threads started for it.

    The threads belong to one batch, so a call never waits for work that other
    callers queued, and a call may start a batch of its own. Each call sees a
    copy of the context variables of the thread that called `run`.

    Warning:
        A daemon thread does not keep the interpreter from exiting: a call still
        running at exit is stopped without completing.
    """

    def __init__(self, *, limit: int | None = None) -> None:
        """Initialize the executor.

        Args:
            limit: The maximum number of calls running at once, or `None` for
                   no limit.

        Raises:
            ValueError: If `limit` is less than one.
        """
        super().__init__()
        if limit is not None and limit < 1:
            msg = f"The limit must be at least one: {limit}"
            raise ValueError(msg)
        self._limit = limit

    @override
    def _run_items(
        self,
        items: list[WorkItem],
        store: Callable[[int, Any], None],
        abort_signal: AbortSignal | None,
    ) -> None:
        done: queue.SimpleQueue[Future[Any] | None] = queue.SimpleQueue()
        futures, jobs = _prepare_jobs(items, done, abort_signal)
        wake = partial(done.put, None)
        if abort_signal is not None:
            abort_signal.add_callback(wake)
        try:
            threads = len(jobs) if self._limit is None else min(self._limit, len(jobs))
            for _ in range(threads):
                threading.Thread(
                    target=_take_jobs, args=(jobs, abort_signal), daemon=True
                ).start()
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
                if not future.cancel():
                    future.add_done_callback(partial(_log_left_running, Future.result))


def _prepare_jobs(
    items: list[WorkItem],
    done: queue.SimpleQueue[Future[Any] | None],
    abort_signal: AbortSignal | None,
) -> tuple[dict[Future[Any], int], deque[_Job]]:
    futures: dict[Future[Any], int] = {}
    jobs: deque[_Job] = deque()
    for index, item in enumerate(items):
        future: Future[Any] = Future()
        future.add_done_callback(done.put)
        futures[future] = index
        # Copied here, on the calling thread, and once per job: a context can
        # be entered by one thread at a time.
        context = contextvars.copy_context()
        job = partial(context.run, _run_as_child, item, abort_signal)
        jobs.append((future, job))
    return futures, jobs


def _take_jobs(jobs: deque[_Job], abort_signal: AbortSignal | None) -> None:
    while True:
        try:
            future, job = jobs.popleft()
        except IndexError:
            return
        if abort_signal is not None and abort_signal.aborting:
            future.cancel()
            continue
        if not future.set_running_or_notify_cancel():
            continue
        try:
            future.set_result(job())
        except BaseException as exc:  # ruff: ignore[blind-except]
            # Delivered to the caller, which is where an exit or an interrupt
            # means something.
            future.set_exception(exc)


def _run_as_child(
    item: WorkItem,
    abort_signal: AbortSignal | None,
) -> Any:  # ruff: ignore[any-type]
    with nullcontext() if abort_signal is None else abort_signal.as_parent():
        return _run_item(item.function, item.args, item.kwargs)
