"""This module implements the default result_handler event handler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import EventHandler

if TYPE_CHECKING:
    from collections.abc import Callable

    from ropt.enums import EnOptEventType
    from ropt.events import EnOptEvent


class CallbackHandler(EventHandler):
    """The default event handler for observing events.

    This event handler listens for events of matching types and forwards them
    to a callback function.

    If the callback performs blocking operations (file I/O, network calls,
    etc.), register this handler with `run_in_thread=True` on the
    [`EventDispatcher`][ropt.workflow.event_handlers.EventDispatcher]:

    ```python
    event_dispatcher.add_event_handler(handler, run_in_thread=True)
    ```
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

    def handle_event(self, event: EnOptEvent) -> None:
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
