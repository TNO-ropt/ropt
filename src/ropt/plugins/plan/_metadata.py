"""This module implements the default setvar step."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Set

from pydantic import BaseModel, ConfigDict

from ropt.enums import EventType
from ropt.plugins.plan.base import EventHandler

if TYPE_CHECKING:
    from ropt.config.plan import EventHandlerConfig
    from ropt.plan import Event, Plan


class DefaultMetadataWith(BaseModel):
    """Parameters used by the default metadata event handler.

    Attributes:
        metadata: Data to set into the metadata
        filter:   Optional tags of the sources to track
    """

    metadata: Dict[str, Any]
    filter: Set[str] = set()

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        arbitrary_types_allowed=True,
    )


class DefaultMetadataHandler(EventHandler):
    """The default metadata event handler."""

    def __init__(self, config: EventHandlerConfig, plan: Plan) -> None:
        """Initialize a default metadata event handler.

        Args:
            config: The configuration of the step
            plan:   The plan that runs this step
        """
        super().__init__(config, plan)

        self._with = DefaultMetadataWith.model_validate(
            config.with_ if "metadata" in config.with_ else {"metadata": config.with_}
        )

    def handle_event(self, event: Event) -> Event:
        """Handle an event.

        Args:
            event: The event to handle

        Returns:
            The (possibly modified) event.
        """
        if (
            event.event_type
            in {
                EventType.FINISHED_EVALUATION,
                EventType.FINISHED_EVALUATOR_STEP,
            }
            and event.results is not None
            and (not self._with.filter or (event.tags & self._with.filter))
        ):
            for results in event.results:
                for key, expr in self._with.metadata.items():
                    results.metadata[key] = self.plan.parse_value(expr)
        return event
