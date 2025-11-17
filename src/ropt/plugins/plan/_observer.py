"""This module implements the default tracker event handler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ropt.plugins.plan.base import EventHandler

if TYPE_CHECKING:
    from collections.abc import Callable

    from ropt.enums import EventType
    from ropt.optimization import Event


class DefaultObserverHandler(EventHandler):
    """The default event handler for observing events.

    This event handler listens for events emitted by specified steps and
    forwards them to one or more callback functions.
    """

    def __init__(
        self, *, event_types: set[EventType], callback: Callable[[Event], None]
    ) -> None:
        """Initialize a default event handler.

        This event handler responds to events received from specified `sources` (plan
        steps) and calls `callback` if the event type matches `event_types`.

        Args:
            plan:        The parent plan instance.
            event_types: The set of event types  to respond to.
            callback:    The callable to call.
        """
        super().__init__()
        self._event_types = event_types
        self._callback = callback

    def handle_event(self, event: Event) -> None:
        """Handle incoming events from the plan.

        This method processes events emitted by a step.

        If a event containing results is received, and its type equals the
        stored event type, the stored callback is called.

        Args:
            event: The event object emitted by the plan.
        """
        if event.event_type in self._event_types:
            self._callback(event)

    @property
    def event_types(self) -> set[EventType]:
        """Return the event types that are handled.

        Returns:
            A set of event types that are handled.
        """
        return self._event_types
