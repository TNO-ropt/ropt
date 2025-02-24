"""This module implements the default tracker handler."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ropt.enums import EventType
from ropt.plugins.plan.base import ResultHandler

from ._utils import _get_all_results, _get_last_result, _get_set, _update_optimal_result

if TYPE_CHECKING:
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
    """

    def __init__(
        self,
        plan: Plan,
        *,
        what: Literal["best", "last", "all"] = "best",
        constraint_tolerance: float | None = 1e-10,
        tags: str | set[str] | None = None,
    ) -> None:
        """Initialize a default tracker results handler object.

        The `what` parameter determines which result is tracked:

        - `"best"`: Tracks the best result added.
        - `"last"`: Tracks the last result added.
        - `"all"`:  Store a tuple with all results.

        If `constraint_tolerance` is set, results that exceed this tolerance on
        constraint values are not tracked.

        The `tags` field allows optional labels to be attached to each result,
        assisting result handlers in filtering relevant results.

        Args:
            plan:                 The plan that runs this step.
            tags:                 Tags to filter the sources to track.
            what:                 Specifies the type of result to store.
            constraint_tolerance: An optional constraint tolerance level.
        """
        super().__init__(plan)
        self._what = what
        self._constraint_tolerance = constraint_tolerance
        self._tags = _get_set(tags)
        self["results"] = None
        self["variables"] = None

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
            and (event.tags & self._tags)
        ):
            results = event.data["results"]
            transformed_results = event.data.get("transformed_results", results)
            filtered_results: FunctionResults | tuple[FunctionResults, ...] | None = (
                None
            )
            match self._what:
                case "all":
                    filtered_results = _get_all_results(
                        results,
                        transformed_results,
                        self._constraint_tolerance,
                    )
                case "best":
                    filtered_results = _update_optimal_result(
                        self["results"],
                        results,
                        transformed_results,
                        self._constraint_tolerance,
                    )
                case "last":
                    filtered_results = _get_last_result(
                        results,
                        transformed_results,
                        self._constraint_tolerance,
                    )
            if filtered_results is not None:
                self["results"] = filtered_results
                variables = (
                    (item.evaluations.variables for item in filtered_results)
                    if isinstance(filtered_results, tuple)
                    else filtered_results.evaluations.variables
                )
                self["variables"] = variables
        return event
