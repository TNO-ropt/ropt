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
        step_name:  Optional name of the step emitting the event
    """

    event_type: EventType
    config: EnOptConfig
    results: Optional[Tuple[Results, ...]] = None
    exit_code: Optional[OptimizerExitCode] = None
    step_name: Optional[str] = None


class EventBroker:
    """A class for handling events.

    An `EventBroker` object is responsible for registering callback objects and
    calling them in response to events that occur during optimization.
    """

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

        Observer functions will be called during optimization if an event of the
        given type occurs. The callable must accept an argument of the
        [`Event`][ropt.optimization.Event] class that contains information about
        the event that occurred.

        Args:
            event:    The type of events to react to
            callback: The function to call if the event is received
        """
        self._subscribers[event].append(callback)

    def emit(
        self,
        event_type: EventType,
        config: EnOptConfig,
        /,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Emit an event of the given type with given data.

        When called, an [`Event`][ropt.optimization.Event] object is constructed
        using the given `event_type` and `config` for its mandatory fields. When
        given, the additional keyword arguments are also passed to the
        [`Event`][ropt.optimization.Event] constructor to set the optional
        fields. All callbacks for the given event type, that were added by the
        `add_observer` method are then called using the newly constructed event
        object as their argument.

        Args:
            event_type: The type of event to emit
            config:     Optimization configuration used by the emitting object
            kwargs:     Keyword arguments used to create an optimization event
        """
        event = Event(event_type, config, **kwargs)
        for callback in self._subscribers[event_type]:
            callback(event)
