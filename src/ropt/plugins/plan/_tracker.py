"""This module implements the default tracker handler."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

from ropt.config.validated_types import ItemOrSet  # noqa: TCH001
from ropt.enums import EventType
from ropt.plugins.plan.base import ResultHandler

from ._utils import _get_all_results, _get_last_result, _update_optimal_result

if TYPE_CHECKING:
    from ropt.config.plan import ResultHandlerConfig
    from ropt.plan import Event, Plan
    from ropt.results import FunctionResults


class DefaultTrackerHandler(ResultHandler):
    """The default tracker results handler object.

    This handler tracks the
    [`Results`][ropt.results.Results] objects that it receives and selects one
    to retain in a variable. Currently it tracks either the last result it
    receives, or the best result. The best result is defined as the result that
    has the lowest weighted objective value. Optionally, results may be filtered
    by checking for violations of constraints, by comparing constraint values to
    a threshold.

    The tracker step uses the [`DefaultTrackerHandlerWith`]
    [ropt.plugins.plan._tracker.DefaultTrackerHandler.DefaultTrackerHandlerWith]
    configuration class to parse the `with` field of the
    [`ResultHandler`][ropt.config.plan.ResultHandlerConfig] used to specify this
    handler in a plan configuration.
    """

    class DefaultTrackerHandlerWith(BaseModel):
        """Parameters for the tracker results handler.

        The tracker stores the tracked results in the variable specified by the
        `var` field. The `type` parameter determines which result is tracked:

        - `"best"`: Tracks the best result added.
        - `"last"`: Tracks the last result added.
        - `"all"`:  Store a tuple with all results.

        If `constraint_tolerance` is set, results that exceed this tolerance on
        constraint values are not tracked.

        The `tags` field allows optional labels to be attached to each result,
        assisting result handlers in filtering relevant results.

        Attributes:
            var:                  The name of the variable to store the tracked result.
            tags:                 Tags to filter the sources to track.
            type:                 Specifies the type of result to store.
            constraint_tolerance: An optional constraint tolerance level.
        """

        var: str
        tags: ItemOrSet[str]
        type_: Literal["best", "last", "all"] = Field(default="best", alias="type")
        constraint_tolerance: Optional[float] = 1e-10

        model_config = ConfigDict(
            extra="forbid",
            validate_default=True,
            frozen=True,
        )

    def __init__(self, config: ResultHandlerConfig, plan: Plan) -> None:
        """Initialize a default tracker results handler object.

        Args:
            config: The configuration of the step.
            plan:   The plan that runs this step.
        """
        super().__init__(config, plan)
        self._with = self.DefaultTrackerHandlerWith.model_validate(config.with_)
        self.plan[self._with.var] = None

    def handle_event(self, event: Event) -> Event:
        """Handle an event.

        Args:
            event: The event to handle.

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
            results: Union[Optional[FunctionResults], Tuple[FunctionResults, ...]]
            results = None
            if self._with.type_ == "all":
                results = _get_all_results(
                    event.results, self._with.constraint_tolerance
                )
                self.plan[self._with.var] = deepcopy(results)
            elif self._with.type_ == "best":
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
