"""The events a compute step emits at lifecycle milestones.

An [`EnOptEvent`][ropt.events.EnOptEvent] carries the event type, the context of
the run, and any results produced. Event handlers consume these to track
progress, store results, or stop the run. See
[`EnOptEventType`][ropt.enums.EnOptEventType] for the available types, and
[Optimization Workflows](../advanced/workflows.md) for usage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ropt.components.compute_steps.base import ComputeStep
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
        source:      The compute step that emitted the event.

    A handler may call `source.stop()` to stop the run that emitted the event.

    See [Optimization Workflows](../advanced/workflows.md#the-enoptevent-object)
    for a detailed description of events and their lifecycle.
    """

    event_type: EnOptEventType
    context: EnOptContext
    results: tuple[Results, ...] = field(default_factory=tuple)
    source: ComputeStep[Any] | None = None
