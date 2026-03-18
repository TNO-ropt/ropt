"""The optimization event class."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ropt.config import EnOptConfig
    from ropt.enums import EnOptEventType
    from ropt.results import Results


@dataclass(slots=True)
class EnOptEvent:
    """Stores data related to an optimization event.

    During the execution of an optimization workflow, events are triggered to
    signal specific occurrences. Callbacks can be registered to react to these
    events and will receive an `EnOptEvent` object containing relevant
    information.

    Attributes:
        event_type: The type of event that occurred.
        config:     The optimizer configuration associated with the event.
        results:    A tuple containing results.
    """

    event_type: EnOptEventType
    config: EnOptConfig
    results: tuple[Results, ...] = field(default_factory=tuple)
