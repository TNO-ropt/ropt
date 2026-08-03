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

    See [Optimization Workflows](../usage/workflows.md#eventforwardhandler) for usage.
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
        """Forward the event to the EventDispatcher.

        Args:
            event: The event to forward.
        """
        self._dispatcher.put_event(event)

    @property
    def event_types(self) -> set[EnOptEventType]:
        """The event types that are handled."""
        return self._event_types
