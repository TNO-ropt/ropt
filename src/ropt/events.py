"""The optimization event class."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ropt.enums import EventType


@dataclass(slots=True)
class Event:
    """Stores data related to an optimization event.

    During the execution of an optimization workflow, events are triggered to
    signal specific occurrences. Callbacks can be registered to react to these
    events and will receive an `Event` object containing relevant information.

    The specific data within the `Event` object varies depending on the event
    type. See the [`EventType`][ropt.enums.EventType] documentation for details.

    Attributes:
        event_type: The type of event that occurred.
        data:       A dictionary containing additional event-specific data.
    """

    event_type: EventType
    data: dict[str, Any] = field(default_factory=dict)
