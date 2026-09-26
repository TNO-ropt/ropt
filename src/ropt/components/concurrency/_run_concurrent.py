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
) -> list[_T]:
    """Run blocking jobs concurrently on dedicated threads and collect results.

    Each job runs on its own thread, so the number of jobs that run at once is
    not capped by any shared thread pool; `limit` optionally bounds how many
    run simultaneously. The first job to raise makes its exception propagate as
    soon as it is observed (fail-fast): jobs that have not started yet are
    skipped, while any already running are abandoned, since a Python thread
    cannot be stopped from the outside.

    Args:
        jobs:  The zero-argument callables to run, one result each.
        limit: The maximum number to run at once, or `None` for no limit.

    Returns:
        The job results, in the order of `jobs`.
    """
    count = len(jobs)
    if count == 0:
        return []

    results: list[_T | None] = [None] * count
    gate = threading.Semaphore(count if limit is None else max(limit, 1))
    stop = threading.Event()
    condition = threading.Condition()
    remaining = count
    first_error: Exception | None = None

    def _worker(index: int, job: Callable[[], _T]) -> None:
        nonlocal remaining, first_error
        with gate:
            if not stop.is_set():
                try:
                    results[index] = job()
                except Exception as exc:  # ruff: ignore[blind-except]
                    with condition:
                        if first_error is None:
                            first_error = exc
                    stop.set()
        with condition:
            remaining -= 1
            condition.notify()

    for index, job in enumerate(jobs):
        threading.Thread(target=_worker, args=(index, job), daemon=True).start()

    # Return once every job has finished, or raise as soon as one fails, without
    # waiting for the still-running siblings that fail-fast leaves abandoned.
    with condition:
        while remaining > 0 and first_error is None:
            condition.wait()
        if first_error is not None:
            raise first_error

    return cast("list[_T]", results)
