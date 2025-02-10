"""This module implements the default tracker handler."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from ropt.config.validated_types import ItemOrSet  # noqa: TC001
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
        constraint_tolerance: float | None = 1e-10

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
            and "results" in event.data
            and (event.tags & self._with.tags)
        ):
            results = event.data["results"]
            transformed_results = event.data.get("transformed_results", results)
            filtered_results: FunctionResults | tuple[FunctionResults, ...] | None = (
                None
            )
            match self._with.type_:
                case "all":
                    filtered_results = _get_all_results(
                        event.config,
                        results,
                        transformed_results,
                        self._with.constraint_tolerance,
                    )
                    self.plan[self._with.var] = deepcopy(filtered_results)
                case "best":
                    filtered_results = _update_optimal_result(
                        event.config,
                        self.plan[self._with.var],
                        results,
                        transformed_results,
                        self._with.constraint_tolerance,
                    )
                case "last":
                    filtered_results = _get_last_result(
                        event.config,
                        results,
                        transformed_results,
                        self._with.constraint_tolerance,
                    )
            if filtered_results is not None:
                self.plan[self._with.var] = deepcopy(filtered_results)
        return event
