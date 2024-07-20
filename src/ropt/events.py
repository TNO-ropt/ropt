"""The optimization event class."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from ropt.enums import EventType, OptimizerExitCode

if TYPE_CHECKING:
    from ropt.config.enopt import EnOptConfig
    from ropt.results import Results


@dataclass
class OptimizationEvent:
    """The `OptimizationEvent` class stores optimization event data.

    While running an optimization, callbacks can be connected to react to events
    triggered during the execution of the optimization plan. These callbacks
    accept a single `OptimizationEvent` object containing information about the
    event.

    The actual data contained in the object depends on the nature of the event.
    Refer to the documentation of the [`EventType`][ropt.enums.EventType]
    enumeration for more details.

    Attributes:
        event_type: The type of the event
        config:     The current configuration object of the executing plan
        results:    Optional results passed with the event
        exit_code:  An optional exit code
    """

    event_type: EventType
    config: EnOptConfig
    results: Optional[Tuple[Results, ...]] = None
    exit_code: Optional[OptimizerExitCode] = None
    step_name: Optional[str] = None


class OptimizationEventBroker:
    """A class for handling optimization events."""

    def __init__(self) -> None:
        """Initialize the optimization event broker."""
        self._subscribers: Dict[
            EventType, List[Callable[[OptimizationEvent], None]]
        ] = {event: [] for event in EventType}

    def add_observer(
        self,
        event: EventType,
        callback: Callable[[OptimizationEvent], None],
    ) -> None:
        """Add an observer function.

        Args:
            event:    The type of events to react to
            callback: The function to call if the event is received
        """
        self._subscribers[event].append(callback)

    def emit(self, event_type: EventType, /, **kwargs: Any) -> None:  # noqa: ANN401
        """Emit an optimization event.

        The keyword arguments are used to construct an
        [`OptimizationEvent`][ropt.events.OptimizationEvent] object of the type
        given by `event_type`. All stored callbacks that react to this event
        type are then called with that event object as their arugment.

        Args:
            event_type: The type of event to emit
            kwargs:     Keyword arguments used to create an optimization event
        """
        event = OptimizationEvent(event_type=event_type, **kwargs)
        for callback in self._subscribers[event_type]:
            callback(event)
