"""Deliver events from a single optimization or evaluation run."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ropt._handlers import EventHandler

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ropt.events import EnOptEvent


def validate_handlers(handlers: Sequence[EventHandler]) -> None:
    for handler in handlers:
        if not isinstance(handler, EventHandler):
            msg = f"Not an event handler: {type(handler).__name__}"
            raise TypeError(msg)


def emit_event(
    event: EnOptEvent,
    handlers: Sequence[EventHandler],
    stop: Callable[[], None],
) -> None:
    event.stop = stop
    for handler in handlers:
        if event.event_type in handler.event_types:
            handler.handle_event(event)
