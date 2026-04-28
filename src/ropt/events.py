"""Optimization event data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ropt.context import EnOptContext
    from ropt.enums import EnOptEventType
    from ropt.results import Results


@dataclass(slots=True)
class EnOptEvent:
    """Container for data emitted with optimization workflow events.

    Events are raised during optimization to signal lifecycle milestones and
    intermediate outcomes. Registered callbacks receive an EnOptEvent instance
    with event metadata, the current context, and any associated results.

    Attributes:
        event_type:  Type of event that occurred.
        context:     Optimizer context associated with the event.
        results:     Tuple of result objects associated with the event.
    """

    event_type: EnOptEventType
    context: EnOptContext
    results: tuple[Results, ...] = field(default_factory=tuple)
