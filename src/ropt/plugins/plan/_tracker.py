"""This module defines the abstract base class for optimization steps."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Literal, Optional, Set

from pydantic import BaseModel, ConfigDict, Field

from ropt.enums import EventType
from ropt.plugins.plan.base import ContextObj

from ._utils import _get_last_result, _update_optimal_result

if TYPE_CHECKING:
    from ropt.config.plan import ContextConfig
    from ropt.optimization import Event
    from ropt.plan import Plan


class DefaultTrackerWith(BaseModel):
    """Parameters for the tracker context object.

    The `type` parameter determines what result is tracked:
    - "optimal": Track the best result added
    - "last": Track the last result added

    Attributes:
        type:                 The type of result to store
        constraint_tolerance: Optional constraint tolerance
    """

    type_: Literal["optimal", "last"] = Field(default="optimal", alias="type")
    constraint_tolerance: Optional[float] = 1e-10
    filter: Set[str] = set()

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
    )


class DefaultTrackerContext(ContextObj):
    """The default tracker context object."""

    def __init__(self, config: ContextConfig, plan: Plan) -> None:
        """Initialize a default tracker context object.

        Args:
            config: The configuration of the step
            plan:   The plan that runs this step
        """
        super().__init__(config, plan)
        self._with = (
            DefaultTrackerWith()
            if config.with_ is None
            else DefaultTrackerWith.model_validate(config.with_)
        )
        self.plan[self.context_config.id] = None
        self.plan.optimizer_context.events.add_observer(
            EventType.FINISHED_EVALUATION, self._track_results
        )
        self.plan.optimizer_context.events.add_observer(
            EventType.FINISHED_EVALUATOR_STEP, self._track_results
        )

    def _track_results(self, event: Event) -> None:
        if self._with.filter and not (event.tags & self._with.filter):
            return
        if event.results is not None:
            results = None
            if self._with.type_ == "optimal":
                results = _update_optimal_result(
                    self.plan[self.context_config.id],
                    event.results,
                    self._with.constraint_tolerance,
                )
            elif self._with.type_ == "last":
                results = _get_last_result(
                    event.results, self._with.constraint_tolerance
                )
            if results is not None:
                self.plan[self.context_config.id] = deepcopy(results)
