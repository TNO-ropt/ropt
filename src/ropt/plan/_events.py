"""The optimization event class."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ropt.enums import EventType


@dataclass(slots=True)
class Event:
    """The `Event` class stores optimization event data.

    While running an optimization plan, callbacks can be connected to react to
    events triggered during execution. These callbacks accept a single `Event`
    object containing information about the event.

    The actual data contained in the object depends on the nature of the event.
    Refer to the documentation of the [`EventType`][ropt.enums.EventType]
    enumeration for more details.

    Attributes:
        event_type: The type of the event.
        tags:       Optional tags added to the results.
        plan_id:    The ID of the plan that generated the event.
        data:       Optional data passed with the event.
    """

    event_type: EventType
    tags: set[str] = field(default_factory=set)
    plan_id: tuple[int, ...] = field(default_factory=tuple)
    data: dict[str, Any] = field(default_factory=dict)
