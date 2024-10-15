"""This module implements the default setvar step."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Dict

from pydantic import BaseModel, ConfigDict

from ropt.config.utils import StrOrSet  # noqa: TCH001
from ropt.enums import EventType
from ropt.plugins.plan.base import ResultHandler

if sys.version_info >= (3, 11):
    pass
else:
    pass


if TYPE_CHECKING:
    from ropt.config.plan import ResultHandlerConfig
    from ropt.plan import Event, Plan


class DefaultMetadataWith(BaseModel):
    """Parameters used by the default metadata results handler.

    Attributes:
        data: Data to set into the metadata
        tags: Tags of the sources to track
    """

    data: Dict[str, Any]
    tags: StrOrSet

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        arbitrary_types_allowed=True,
    )


class DefaultMetadataHandler(ResultHandler):
    """The default metadata results handler."""

    def __init__(self, config: ResultHandlerConfig, plan: Plan) -> None:
        """Initialize a default metadata results handler.

        Args:
            config: The configuration of the step
            plan:   The plan that runs this step
        """
        super().__init__(config, plan)

        self._with = DefaultMetadataWith.model_validate(config.with_)

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
            and (event.tags & self._with.tags)
        ):
            for results in event.results:
                for key, expr in self._with.data.items():
                    if not isinstance(expr, str):
                        results.metadata[key] = expr
                    elif expr.startswith("$") and expr[1:] in self._plan:
                        results.metadata[key] = self._plan[expr[1:]]
                    else:
                        results.metadata[key] = self.plan.interpolate_string(expr)
        return event
