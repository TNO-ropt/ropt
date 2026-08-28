"""Checks that turn a misused pool or handler group into a clear message.

A pool or group whose session has closed is still a live object, so without a
check here the run would get an
[`ExecutorStopped`][ropt.exceptions.ExecutorStopped] from inside its first
evaluation, or silently drop its events.

The checks run once, at the entry point, before any work starts. A pool that
dies *while* a run is using it is a different case: that run is already going,
and ends the way it always has.

Two checks are not here. Work submitted to the pool it is already running on is
refused by the executor, at submit time, that being the only point nested runs
and [`offload`][ropt.simple.offload] both pass through. A pool carried into a
worker is refused earlier still: it cannot be serialized, so the submission that
carried it fails before any worker sees it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ropt.exceptions import WorkflowError

from ._handlers import SharedHandlers

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ropt.components.event_handlers import EventHandler

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


def check_handlers(
    handlers: Sequence[EventHandler | SharedHandlers] | None,
) -> None:
    """Reject a handler group that cannot receive a new run's results.

    Args:
        handlers: The handlers and groups the run was given, if any.

    Raises:
        WorkflowError: If a group is closed.
    """
    for item in handlers or ():
        if isinstance(item, SharedHandlers) and item.closed:
            msg = (
                "This group of shared handlers is closed and cannot take new "
                "runs; its session has ended, or it was closed directly. Group "
                "the handlers again on an open session."
            )
            raise WorkflowError(msg)
