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

    Attributes:
        event_type:  Type of event that occurred.
        context:     Optimizer context associated with the event.
        results:     Tuple of result objects associated with the event.

    See [Optimization Workflows](../usage/workflows.md#the-enoptevent-object)
    for a detailed description of events and their lifecycle.
    """

    event_type: EnOptEventType
    context: EnOptContext
    results: tuple[Results, ...] = field(default_factory=tuple)
