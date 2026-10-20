"""The events an optimization or evaluation emits at its lifecycle milestones.

An [`EnOptEvent`][ropt.events.EnOptEvent] carries the event type, the
configuration in use, and any results produced. Event handlers consume these
to track progress, store results, or stop the optimization. See
[`EnOptEventType`][ropt.enums.EnOptEventType] for the available types, and
[Handling Results](../results/handlers.md#writing-your-own-handler) for usage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from ropt.context import EnOptContext
    from ropt.enums import EnOptEventType
    from ropt.results import Results


@dataclass(slots=True)
class EnOptEvent:
    """Container for the data emitted with an event.

    Attributes:
        event_type: Type of event that occurred.
        context:    The validated configuration in use.
        results:    Tuple of result objects associated with the event.
        stop:       Stops the optimization that emitted the event.

    See [Handling Results](../results/handlers.md#writing-your-own-handler) for
    when each event fires and what it carries.
    """

    event_type: EnOptEventType
    context: EnOptContext
    results: tuple[Results, ...] = field(default_factory=tuple)
    stop: Callable[[], None] | None = None
