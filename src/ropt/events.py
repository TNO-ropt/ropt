"""The optimization event class."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

from ropt.enums import EventType, OptimizerExitCode  # noqa: TCH001

if TYPE_CHECKING:
    from ropt.config.enopt import EnOptConfig
    from ropt.results import Results


@dataclass
class OptimizationEvent:
    """The `OptimizationEvent` class stores optimization event data.

    While running an optimization, callbacks can be connected to react to events
    triggered during the execution of the plan. These callbacks accept a single
    `OptimizationEvent` object containing information about the event.

    The actual data contained in the object depends on the nature of the event.
    Refer to the documentation of the [`EventType`][ropt.enums.EventType]
    enumeration for more details.

    Attributes:
        event_type: The type of the event.
        config:     The current configuration object of the executing plan.
        results:    Optional results passed with the event.
        exit_code:  An optional exit code.
    """

    event_type: EventType
    config: EnOptConfig
    results: Optional[Tuple[Results, ...]] = None
    exit_code: Optional[OptimizerExitCode] = None
