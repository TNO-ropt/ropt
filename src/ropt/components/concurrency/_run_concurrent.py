"""A primitive for running blocking jobs concurrently.

The drivers of `optimize_many` each block for the whole of one run, waiting for
evaluations that the executor runs elsewhere. On a shared, bounded thread pool
they would occupy every slot and the work they wait for would queue behind
them. `run_concurrent` gives each job a thread of its own, so the number of
jobs that run at once is capped only by its own `limit`.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, TypeVar, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_T = TypeVar("_T")


def run_concurrent(
    jobs: Sequence[Callable[[], _T]], limit: int | None = None
) -> list[_T | BaseException]:
    """Run blocking jobs concurrently on dedicated threads and collect outcomes.

    Each job runs on its own thread, so the number of jobs that run at once is
    not capped by any shared thread pool; `limit` optionally bounds how many
    run simultaneously. A job that raises does not affect its siblings: every
    job is run, and the exception it raised is returned in its place. What a
    failure means is left to the caller.

    Args:
        jobs:  The zero-argument callables to run, one outcome each.
        limit: The maximum number to run at once, or `None` for no limit.

    Returns:
        Per job, in the order of `jobs`, its result or the exception it raised.
    """
    count = len(jobs)
    if count == 0:
        return []

    outcomes = cast("list[_T | BaseException]", [None] * count)
    gate = threading.Semaphore(count if limit is None else max(limit, 1))

    def _worker(index: int, job: Callable[[], _T]) -> None:
        with gate:
            try:
                outcomes[index] = job()
            except BaseException as exc:  # ruff: ignore[blind-except]
                outcomes[index] = exc

    threads = [
        threading.Thread(target=_worker, args=(index, job), daemon=True)
        for index, job in enumerate(jobs)
    ]
    for thread in threads:
        thread.start()
    # Daemon threads, so an interrupt that breaks these joins abandons what is
    # still running instead of holding up the interpreter.
    for thread in threads:
        thread.join()

    return outcomes
