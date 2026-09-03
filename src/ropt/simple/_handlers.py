"""Result handlers: local to one run, or shared by a group of runs.

Every entry point takes `handlers=`, a list that may mix two kinds of item:

- an [`EventHandler`][ropt.components.event_handlers.EventHandler] is **local**.
  It is claimed for the duration of the run, so it belongs to that run alone,
  and released afterwards, so a later run can reuse it.
- a [`SharedHandlers`][ropt.simple.SharedHandlers] group is **shared**. The run
  forwards its events to the group's dispatcher, which serializes them across
  every run feeding it, so its handlers accumulate results without locking.

A group is built on a session with
[`shared_handlers`][ropt.simple.Session.shared_handlers] and passed to runs
explicitly, exactly like a pool.

See [Result Handlers](../running/handlers.md) for the local-vs-shared
distinction and lifecycle.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Self

from ropt.components.event_handlers import (
    EventDispatcher,
    EventForwardHandler,
    EventHandler,
)
from ropt.exceptions import WorkflowError

from ._report import make_report_handler

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from ropt.components.compute_steps import ComputeStep
    from ropt.enums import EnOptEventType

    from ._report import ReportCallback
    from ._session import _Session


_IN_USE = (
    "This handler is already in use and cannot join a group of shared handlers. "
    "A handler passed to a run in `handlers=` stays bound to that run, and one "
    "already held by another group cannot be shared twice. Use a separate "
    "handler here."
)

_LISTED_TWICE = (
    "This handler is listed more than once in the same group of shared "
    "handlers. List it once."
)


class SharedHandlers:
    """A group of result handlers that several runs share.

    Created by [`shared_handlers`][ropt.simple.Session.shared_handlers] and
    passed to the runs that should feed it. Each handler in the group sees the
    results of every one of those runs, sequential or concurrent, and the
    group's dispatcher serializes them, so a handler that accumulates across
    runs needs no locking of its own.

    A group lives until its session closes, which releases it and its handlers.
    Release it earlier with [`close`][ropt.simple.SharedHandlers.close], or by
    using it as a context manager. See
    [Running Optimizations](../running/running.md) for a walkthrough.
    """

    def __init__(
        self, entries: Sequence[tuple[EventHandler, bool]], session: _Session
    ) -> None:
        """Initialize the group and start its dispatcher.

        Args:
            entries: The handlers, each with whether to run it in a thread.
            session: The session whose event loop the dispatcher runs on.
        """
        self._session = session
        self._dispatcher = EventDispatcher()
        self._handlers: list[EventHandler] = []
        self._closed = False
        try:
            self._claim(entries)
            # Registered before starting, so a session that shuts down during
            # the start still has the group to close.
            session.add_extra(self)
            session.open_dispatcher(self._dispatcher)
        except BaseException:
            session.discard_extra(self)
            self._release()
            raise

    @property
    def closed(self) -> bool:
        """Whether the group has been released.

        Returns:
            `True` once the group, or the session that built it, was closed.
        """
        return self._closed

    def attach_to(self, step: ComputeStep[Any]) -> None:
        """Forward the events of a run's compute step to this group's handlers.

        A fresh forwarding handler is added per step (one handler cannot serve
        several steps), carrying only the event types the group's handlers want.

        Args:
            step: The compute step whose events feed the shared handlers.
        """
        event_types: set[EnOptEventType] = set()
        for handler in self._handlers:
            event_types |= handler.event_types
        # Only what some handler in the group asked for: forwarding blocks the
        # run until the dispatcher is done, so events nobody wants cost it time.
        if event_types:
            step.add_event_handler(
                EventForwardHandler(self._dispatcher, event_types=event_types)
            )

    def close(self) -> None:
        """Release the group's handlers without waiting for the session to close.

        See [Sharing a handler across concurrent runs](../running/handlers.md#sharing-a-handler-across-concurrent-runs)
        for when this matters and the resulting lifecycle.
        """
        if self._closed:
            return
        self._closed = True
        self._session.discard_extra(self)
        try:
            # Before cancelling, so the dispatcher is still there to remove them
            # from; anything queued has already been handled, because
            # `dispatch_event` blocks its caller until it has.
            self._release()
        finally:
            self._session.close_dispatcher(self._dispatcher)

    def __enter__(self) -> Self:
        """Enter a block that closes the group on exit.

        Returns:
            The group itself.
        """
        return self

    def __exit__(self, *_exc: object) -> None:
        """Close the group."""
        self.close()

    def _claim(self, entries: Sequence[tuple[EventHandler, bool]]) -> None:
        for handler, run_in_thread in entries:
            try:
                self._dispatcher.add_event_handler(handler, run_in_thread=run_in_thread)
            except WorkflowError as exc:
                # The low-level refusal is phrased in terms of dispatchers and
                # compute steps, neither of which this API ever hands out, so
                # both causes are restated here in its own vocabulary.
                message = _LISTED_TWICE if handler in self._handlers else _IN_USE
                raise WorkflowError(message) from exc
            self._handlers.append(handler)

    def _release(self) -> None:
        # Cancelling a dispatcher does not unregister its handlers, so without
        # this they would stay marked as attached and could never be used again.
        handlers, self._handlers = self._handlers, []
        for handler in handlers:
            self._dispatcher.remove_event_handler(handler)


def group_entries(
    handlers: Sequence[EventHandler],
    threaded: EventHandler | Sequence[EventHandler],
    report: ReportCallback | None,
) -> list[tuple[EventHandler, bool]]:
    """Pair each handler of a group with whether it runs in a thread.

    Args:
        handlers: The handlers to run on the session's event-loop thread.
        threaded: The handlers to run on a worker thread instead.
        report:   An optional callback added to the group as a report handler.

    Returns:
        The handlers of the group, each with its threading choice.
    """
    in_thread = (threaded,) if isinstance(threaded, EventHandler) else tuple(threaded)
    entries: list[tuple[EventHandler, bool]] = [
        *((item, False) for item in handlers),
        *((item, True) for item in in_thread),
    ]
    if report is not None:
        entries.append((make_report_handler(report), False))
    return entries


def split_handlers(
    handlers: Sequence[EventHandler | SharedHandlers] | None,
) -> tuple[list[EventHandler], list[SharedHandlers]]:
    """Separate a run's local handlers from the groups it feeds.

    Args:
        handlers: The mixed list passed to an entry point, if any.

    Returns:
        The local handlers, and the shared groups, each in the given order.
    """
    local: list[EventHandler] = []
    groups: list[SharedHandlers] = []
    for item in handlers or ():
        if isinstance(item, SharedHandlers):
            groups.append(item)
        else:
            local.append(item)
    return local, groups


@contextmanager
def attach_handlers(
    step: ComputeStep[Any],
    handlers: Sequence[EventHandler | SharedHandlers] | None,
    report: ReportCallback | None,
) -> Iterator[None]:
    """Wire a run's handlers to its compute step for the duration of the run.

    Local handlers are claimed before any is attached, so a run that cannot have
    all of them leaves every one of them free, and released again afterwards, so
    a later run can reuse them. Shared groups are attached instead of claimed,
    since they are not exclusive to one run.

    Args:
        step:     The compute step of the run.
        handlers: The local handlers and shared groups to wire up.
        report:   An optional callback wired up as a local report handler.

    Yields:
        Nothing; the handlers stay attached for the body of the block.
    """
    local, groups = split_handlers(handlers)
    if report is not None:
        local.append(make_report_handler(report))
    claimed: list[EventHandler] = []
    try:
        # Claimed first, all of them, before anything is attached: a run that
        # cannot have every handler it asked for must leave them all free.
        for handler in local:
            handler.claim()
            claimed.append(handler)
        # Attaching is what binds a handler to compute steps for good, which is
        # why a handler used locally can never join a shared group afterwards.
        for handler in local:
            step.add_event_handler(handler)
        for group in groups:
            group.attach_to(step)
        yield
    finally:
        for handler in claimed:
            handler.release()
