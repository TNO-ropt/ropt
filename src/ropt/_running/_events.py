"""Deliver events from a single optimization or evaluation run."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from ropt._handlers import EventHandler
from ropt.enums import EnOptEventType
from ropt.events import EnOptEvent

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ropt.context import EnOptContext
    from ropt.results import Results


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


def emit_start_evaluation(
    context: EnOptContext,
    handlers: Sequence[EventHandler],
    stop: Callable[[], None],
    default_metadata: dict[str, object],
) -> dict[str, object]:
    event = EnOptEvent(
        event_type=EnOptEventType.START_EVALUATION,
        context=context,
        metadata=deepcopy(default_metadata),
    )
    emit_event(event, handlers, stop)
    return deepcopy(event.metadata)


def emit_finished_evaluation(
    context: EnOptContext,
    handlers: Sequence[EventHandler],
    stop: Callable[[], None],
    results: tuple[Results, ...],
) -> None:
    event = EnOptEvent(
        event_type=EnOptEventType.FINISHED_EVALUATION,
        context=context,
        results=results,
    )
    emit_event(event, handlers, stop)
