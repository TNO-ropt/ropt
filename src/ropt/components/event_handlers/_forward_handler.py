"""This module implements the event forwarding handler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import EventHandler

if TYPE_CHECKING:
    from ropt.enums import EnOptEventType
    from ropt.events import EnOptEvent

    from ._event_dispatcher import EventDispatcher


class EventForwardHandler(EventHandler):
    """Forwards events from a compute step to an `EventDispatcher`.

    See [Optimization Workflows](../workflows/workflows.md#eventforwardhandler) for usage.
    """

    def __init__(
        self,
        dispatcher: EventDispatcher,
        *,
        event_types: set[EnOptEventType],
    ) -> None:
        """Initialize the EventForwardHandler.

        Args:
            dispatcher:  The EventDispatcher to forward events to.
            event_types: The set of event types to forward.
        """
        super().__init__()
        self._dispatcher = dispatcher
        self._event_types = event_types

    def handle_event(self, event: EnOptEvent) -> None:
        """Forward the event to the EventDispatcher and wait for it.

        Submits the event to the dispatcher and blocks on the emitting run's own
        call stack until every handler has processed it. A handler fault is
        re-raised here as a clean exception with a normal exit code, instead of
        tearing down the session task group. See
        [`dispatch_event`][ropt.components.event_handlers.EventDispatcher.dispatch_event].

        Args:
            event: The event to forward.
        """
        self._dispatcher.dispatch_event(event)

    @property
    def event_types(self) -> set[EnOptEventType]:
        """The event types that are handled."""
        return self._event_types
