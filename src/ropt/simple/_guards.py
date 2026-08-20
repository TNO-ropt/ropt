"""Checks that turn a misused pool or handler group into a clear message.

Both mistakes these catch surface far from their cause without a check here. A
pool or group carried into a worker process arrives as an inert placeholder, and
the run would fail on the first attribute it touched. One whose session has
closed is still a live object, and the run would get an
[`ExecutorStopped`][ropt.exceptions.ExecutorStopped] from inside its first
evaluation, or silently drop its events.

The checks run once, at the entry point, before any work starts. A pool that
dies *while* a run is using it is a different case: that run is already going,
and ends the way it always has.

One check is not here: work submitted to the pool it is already running on is
refused by the executor, at submit time, that being the only point nested runs
and [`offload`][ropt.simple.offload] both pass through.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ropt.components._transferred import _Placeholder
from ropt.exceptions import WorkflowError

from ._handlers import SharedHandlers

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ropt.components.event_handlers import EventHandler

    from ._pool import WorkerPool


def _reject_transferred(obj: object) -> None:
    if isinstance(obj, _Placeholder):
        # The placeholder remembers what it stood in for, so the message can
        # name the object the caller actually passed.
        subject = obj._subject  # ruff: ignore[private-member-access]
        msg = (
            f"{subject} cannot be used in a worker process: it belongs to the "
            "session of the run that started this evaluation, which lives in "
            "another process. Do the work in the evaluation function itself, or "
            "return what it needs and act on that in the main process."
        )
        raise WorkflowError(msg)


def check_pool(pool: WorkerPool | None) -> None:
    """Reject a pool that cannot run a new run's evaluations.

    Args:
        pool: The pool the run was given, if any.

    Raises:
        WorkflowError: If the pool was transferred into a worker process, or is
                       closed, or its session has stopped.
    """
    if pool is None:
        return
    _reject_transferred(pool)
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
        WorkflowError: If a group was transferred into a worker process, or is
                       closed.
    """
    for item in handlers or ():
        _reject_transferred(item)
        if isinstance(item, SharedHandlers) and item.closed:
            msg = (
                "This group of shared handlers is closed and cannot take new "
                "runs; its session has ended, or it was closed directly. Group "
                "the handlers again on an open session."
            )
            raise WorkflowError(msg)
