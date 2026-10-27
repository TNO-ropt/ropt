"""This module implements the callback event handler."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from .base import EventHandler

if TYPE_CHECKING:
    from collections.abc import Callable

    from ropt.enums import EnOptEventType
    from ropt.events import EnOptEvent


class CallbackHandler(EventHandler):
    """Forward selected events to a callback.

    The callback receives an [`EnOptEvent`][ropt.events.EnOptEvent], not an
    individual result. Its return value is ignored. See
    [Callback handlers](../results/handlers.md#callbackhandler) for usage.
    """

    def __init__(
        self,
        *,
        event_types: set[EnOptEventType],
        callback: Callable[[EnOptEvent], None],
    ) -> None:
        """Initialize the CallbackHandler.

        Args:
            event_types: The set of event types to respond to.
            callback:    The callable to invoke for matching events.
        """
        super().__init__()
        self._event_types = event_types
        self._callback = callback

    @override
    def _handle_event(self, event: EnOptEvent) -> None:
        if event.event_type in self._event_types:
            self._callback(event)

    @property
    @override
    def event_types(self) -> set[EnOptEventType]:
        """The event types that are handled.

        Returns:
            A set of event types that are handled.
        """
        return self._event_types
