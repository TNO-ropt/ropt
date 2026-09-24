"""Checks that turn a misused pool into a clear message.

A pool whose session has closed is still a live object, so without a check here
the run would get an [`ExecutorStopped`][ropt.exceptions.ExecutorStopped] from
inside its first evaluation.

The checks run once, at the entry point, before any work starts. A pool that
dies *while* a run is using it is a different case: that run is already going,
and ends the way it always has.

Two checks are not here. Work submitted to the pool it is already running on is
refused by the executor, at submit time, that being the only point nested runs
and [`offload`][ropt.simple.offload] both pass through. A serial pool never
reaches that point, having no executor, and needs no refusal either: it has no
workers for a waiting caller to occupy, so a nested run on it evaluates inline
like any other. A pool carried into a worker is refused earlier still: it cannot
be serialized, so the submission that carried it fails before any worker sees
it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ropt.exceptions import WorkflowError

if TYPE_CHECKING:
    from ._pool import WorkerPool


def check_pool(pool: WorkerPool | None) -> None:
    """Reject a pool that cannot run a new run's evaluations.

    Args:
        pool: The pool the run was given, if any.

    Raises:
        WorkflowError: If the pool is closed, or its session has stopped.
    """
    if pool is None:
        return
    if pool.closed:
        msg = (
            "This worker pool is closed and cannot take new runs; its session "
            "has ended, or it was closed directly. Build a pool on an open "
            "session, or run without one to evaluate in-process."
        )
        raise WorkflowError(msg)
