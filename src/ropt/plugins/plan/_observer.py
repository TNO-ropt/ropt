"""This module implements the default tracker event handler."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from ropt.plugins.plan.base import EventHandler

if TYPE_CHECKING:
    import uuid

    from ropt.enums import EventType
    from ropt.plan import Event, Plan


class DefaultObserverHandler(EventHandler):
    """The default event handler for observing events.

    This event handler listens for events emitted by specified `sources` (plan
    steps) and forwards them to one or more callback functions.

    The `sources` parameter filters which events are observed.
    """

    def __init__(
        self,
        plan: Plan,
        *,
        event_types: set[EventType],
        callback: Callable[[Event], None],
        sources: set[uuid.UUID] | None = None,
    ) -> None:
        """Initialize a default event handler.

        This event handler responds to events received from specified `sources` (plan
        steps) and calls `callback` if the event type matches `event_types`.

        The `sources` parameter acts as a filter, determining which plan steps
        this event handler should listen to. It should be a set containing the
        unique IDs (UUIDs) of the `PlanStep` instances whose event you want to
        receive. When an event is received, this event handler checks if the ID
        of the step that emitted the event (`event.source`) is present in the
        `sources` set. If it is, and the event type (event.event_type) is
        present in the `event_types` set, the handler will call `callback`;
        otherwise, it ignores the event. If `sources` is `None`, events from all
        sources will be processed.

        Args:
            plan:        The parent plan instance.
            sources:     Optional set of step UUIDs whose results should be stored.
            event_types: The set of event types  to respond to.
            callback:    The callable to call.
        """
        super().__init__(plan)
        self._sources = sources
        self._event_types = event_types
        self._callback = callback

    def handle_event(self, event: Event) -> None:
        """Handle incoming events from the plan.

        This method processes events emitted by the parent plan. It specifically
        listens for events originating from steps whose IDs are included in the
        `sources` set configured during initialization.

        If a event containing results is received, and its type equals the
        stored event type, the stored callback is called.

        Args:
            event: The event object emitted by the plan.
        """
        if (
            self._sources is None or event.source in self._sources
        ) and event.event_type in self._event_types:
            self._callback(event)
