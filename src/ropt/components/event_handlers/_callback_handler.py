"""This module implements the callback event handler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import EventHandler

if TYPE_CHECKING:
    from collections.abc import Callable

    from ropt.enums import EnOptEventType
    from ropt.events import EnOptEvent


class CallbackHandler(EventHandler):
    """Forwards events of matching types to a callback.

    A callback that blocks (file I/O, network calls) should be registered with
    `run_in_thread=True` on the
    [`EventDispatcher`][ropt.components.event_handlers.EventDispatcher]. See
    [Optimization Workflows](../advanced/workflows.md#event-dispatcher).
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

    def _handle_event(self, event: EnOptEvent) -> None:
        """Handle incoming events.

        Args:
            event: The event object.
        """
        if event.event_type in self._event_types:
            self._callback(event)

    @property
    def event_types(self) -> set[EnOptEventType]:
        """The event types that are handled.

        Returns:
            A set of event types that are handled.
        """
        return self._event_types
