"""The optimization event class."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from ropt.enums import EventType, OptimizerExitCode

if TYPE_CHECKING:
    from ropt.config.enopt import EnOptConfig
    from ropt.results import Results


@dataclass
class Event:
    """The `Event` class stores optimization event data.

    While running an optimization plan, callbacks can be connected to react to
    events triggered during execution. These callbacks accept a single `Event`
    object containing information about the event.

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


class EventBroker:
    """A class for handling events."""

    def __init__(self) -> None:
        """Initialize the event broker."""
        self._subscribers: Dict[EventType, List[Callable[[Event], None]]] = {
            event: [] for event in EventType
        }

    def add_observer(
        self,
        event: EventType,
        callback: Callable[[Event], None],
    ) -> None:
        """Add an observer function.

        Args:
            event:    The type of events to react to
            callback: The function to call if the event is received
        """
        self._subscribers[event].append(callback)

    def emit(self, event_type: EventType, /, **kwargs: Any) -> None:  # noqa: ANN401
        """Emit an event.

        The keyword arguments are used to construct an
        [`Event`][ropt.events.Event] object of the type given by `event_type`.
        All stored callbacks that react to this event type are then called with
        that event object as their argument.

        Args:
            event_type: The type of event to emit
            kwargs:     Keyword arguments used to create an optimization event
        """
        event = Event(event_type=event_type, **kwargs)
        for callback in self._subscribers[event_type]:
            callback(event)
