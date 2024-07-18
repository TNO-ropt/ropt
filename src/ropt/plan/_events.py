"""Event handling code."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from ropt.enums import EventType
from ropt.events import OptimizationEvent


class OptimizationEventBroker:
    def __init__(self) -> None:
        self._subscribers: Dict[
            EventType, List[Callable[[OptimizationEvent], None]]
        ] = {event: [] for event in EventType}

    def add_observer(
        self,
        event: EventType,
        callback: Callable[[OptimizationEvent], None],
    ) -> None:
        self._subscribers[event].append(callback)

    def emit(self, **kwargs: Any) -> None:  # noqa: ANN401
        event = OptimizationEvent(**kwargs)
        for callback in self._subscribers[event.event_type]:
            callback(event)
