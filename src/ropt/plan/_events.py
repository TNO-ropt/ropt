"""The optimization event class."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Set

if TYPE_CHECKING:
    from ropt.config.enopt import EnOptConfig
    from ropt.enums import EventType, OptimizerExitCode
    from ropt.results import Results


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
        config:     The current configuration object of the executing plan.
        results:    Optional results passed with the event.
        exit_code:  An optional exit code.
        tags:       Optional tags added to the results.
        plan_id:    The ID of the plan that generated the event.
    """

    event_type: EventType
    config: EnOptConfig
    results: tuple[Results, ...] | None = None
    exit_code: OptimizerExitCode | None = None
    tags: Set[str] = field(default_factory=set)
    plan_id: tuple[int, ...] = field(default_factory=tuple)
