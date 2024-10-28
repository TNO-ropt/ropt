"""This module defines the abstract base class for optimization steps."""

from __future__ import annotations

import sys
from copy import deepcopy
from typing import TYPE_CHECKING, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from ropt.config.validated_types import ItemOrSet  # noqa: TCH001
from ropt.enums import EventType
from ropt.plugins.plan.base import ResultHandler

from ._utils import _get_last_result, _update_optimal_result

if sys.version_info >= (3, 11):
    pass
else:
    pass


if TYPE_CHECKING:
    from ropt.config.plan import ResultHandlerConfig
    from ropt.plan import Event, Plan


class DefaultTrackerHandler(ResultHandler):
    """The default tracker results handler object."""

    class DefaultTrackerHandlerWith(BaseModel):
        """Parameters for the tracker results handler.

        The `type` parameter determines what result is tracked:
        - "optimal": Track the best result added
        - "last": Track the last result added

        Attributes:
            var:                  The name of the variable to store the tracked result
            tags:                 Tags of the sources to track
            type:                 The type of result to store
            constraint_tolerance: Optional constraint tolerance
        """

        var: str
        tags: ItemOrSet[str]
        type_: Literal["optimal", "last"] = Field(default="optimal", alias="type")
        constraint_tolerance: Optional[float] = 1e-10

        model_config = ConfigDict(
            extra="forbid",
            validate_default=True,
        )

    def __init__(self, config: ResultHandlerConfig, plan: Plan) -> None:
        """Initialize a default tracker results handler object.

        Args:
            config: The configuration of the step
            plan:   The plan that runs this step
        """
        super().__init__(config, plan)
        self._with = self.DefaultTrackerHandlerWith.model_validate(config.with_)
        self.plan[self._with.var] = None

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
            results = None
            if self._with.type_ == "optimal":
                results = _update_optimal_result(
                    self.plan[self._with.var],
                    event.results,
                    self._with.constraint_tolerance,
                )
            elif self._with.type_ == "last":
                results = _get_last_result(
                    event.results, self._with.constraint_tolerance
                )
            if results is not None:
                self.plan[self._with.var] = deepcopy(results)
        return event
