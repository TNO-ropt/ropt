"""This module implements the event forwarding handler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import EventHandler

if TYPE_CHECKING:
    from ropt.enums import EnOptEventType
    from ropt.events import EnOptEvent
    from ropt.workflow.executors._event_server import EventServer


class EventForwardHandler(EventHandler):
    """Forwards events from a compute step to an `EventServer`.

    See [Optimization Workflows](../usage/workflows.md#eventforwardhandler) for usage.
    """

    def __init__(
        self,
        server: EventServer,
        *,
        event_types: set[EnOptEventType],
    ) -> None:
        """Initialize the EventForwardHandler.

        Args:
            server:      The EventServer to forward events to.
            event_types: The set of event types to forward.
        """
        super().__init__()
        self._server = server
        self._event_types = event_types

    def handle_event(self, event: EnOptEvent) -> None:
        """Forward the event to the EventServer.

        Args:
            event: The event to forward.
        """
        self._server.put_event(event)

    @property
    def event_types(self) -> set[EnOptEventType]:
        """The event types that are handled."""
        return self._event_types
