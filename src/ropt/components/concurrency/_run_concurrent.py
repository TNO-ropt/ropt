"""A primitive for running blocking jobs concurrently.

The drivers of `optimize_many` each block for the whole of one run, waiting for
evaluations that the executor runs elsewhere. On a shared, bounded thread pool
they would occupy every slot and the work they wait for would queue behind
them. `run_concurrent` uses threads of its own, so the number of jobs that run
at once is capped only by its own `limit`.
"""

from __future__ import annotations

import queue
import threading
from typing import TYPE_CHECKING, TypeVar, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_T = TypeVar("_T")


def run_concurrent(
    jobs: Sequence[Callable[[], _T]],
    limit: int | None = None,
    *,
    interrupt: Callable[[], None] | None = None,
) -> list[_T | BaseException]:
    """Run blocking jobs concurrently on dedicated threads and collect outcomes.

    The threads are this call's own, so a job never waits for work queued
    behind it in a thread pool it shares. `limit` bounds how many jobs run at
    once, and bounds the thread count with it: a job awaiting its turn holds no
    thread. A job that raises does not affect its siblings: every job is run,
    and the exception it raised is returned in its place. What a failure means
    is left to the caller.

    An interrupt that breaks the wait abandons the jobs still running. Pass
    `interrupt` to be called at that point: the jobs are then waited for, which
    takes as long as they take to observe it, and a second interrupt abandons
    them after all.

    Args:
        jobs:      The zero-argument callables to run, one outcome each.
        limit:     The maximum number to run at once, or `None` for no limit.
                   A value below 1 is treated as 1.
        interrupt: Called to stop the jobs when an interrupt breaks the wait.

    Returns:
        Per job, in the order of `jobs`, its result or the exception it raised.
    """
    count = len(jobs)
    if count == 0:
        return []

    outcomes = cast("list[_T | BaseException]", [None] * count)
    pending: queue.SimpleQueue[int] = queue.SimpleQueue()
    for index in range(count):
        pending.put(index)

    workers = count if limit is None else min(max(limit, 1), count)
    threads = [
        threading.Thread(target=_consume, args=(jobs, pending, outcomes), daemon=True)
        for _ in range(workers)
    ]
    for thread in threads:
        thread.start()
    try:
        for thread in threads:
            thread.join()
    except BaseException:
        # A thread cannot be interrupted, so the jobs are asked to stop and then
        # waited for. Daemon threads, so a second interrupt breaking this join
        # abandons what is still running instead of holding up the interpreter.
        if interrupt is not None:
            interrupt()
            for thread in threads:
                thread.join()
        raise

    return outcomes


def _consume(
    jobs: Sequence[Callable[[], _T]],
    pending: queue.SimpleQueue[int],
    outcomes: list[_T | BaseException],
) -> None:
    while True:
        try:
            index = pending.get_nowait()
        except queue.Empty:
            return
        try:
            outcomes[index] = jobs[index]()
        except BaseException as exc:  # ruff: ignore[blind-except]
            outcomes[index] = exc
