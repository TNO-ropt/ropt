"""This module implements the default tracker handler."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ropt.enums import EventType
from ropt.plugins.plan.base import ResultHandler

from ._utils import _get_last_result, _update_optimal_result

if TYPE_CHECKING:
    import uuid

    from ropt.plan import Event, Plan
    from ropt.results import FunctionResults
    from ropt.transforms import OptModelTransforms


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
        what: Literal["best", "last"] = "best",
        constraint_tolerance: float | None = None,
        sources: set[uuid.UUID] | None = None,
        transforms: OptModelTransforms | None = None,
    ) -> None:
        """Initialize a default tracker results handler.

        This handler monitors `Results` objects from specified sources and
        selects a single result to retain based on the specified criteria. It
        can track either the best result encountered so far or the most recent
        result. The "best" result is determined by the lowest weighted
        objective value.

        Results can optionally be filtered based on constraint violations. If a
        `constraint_tolerance` is provided, results that violate constraints
        beyond this tolerance will be discarded and not tracked.

        The `sources` parameter allows you to specify which steps' results
        should be tracked. Only results from steps whose IDs are included in
        this set will be considered.

        Tracking of results is based on the results in the optimizer domain, and
        the tracked results are accessible via the `"results"` key. If the
        `transforms` argument is not `None`, tracking still occurs in the
        optimizer domain, but the results accessible via the `"results"` key are
        transformed to the user domain.

        Args:
            plan:                 The plan this handler is part of.
            what:                 Specifies whether to track the "best" or "last" result.
            constraint_tolerance: Constraint tolerance for filtering results.
            sources:              A set of UUIDs of steps whose results to track.
            transforms:           Optional transforms to apply to the final results.
        """
        super().__init__(plan)
        self._what = what
        self._constraint_tolerance = constraint_tolerance
        self._sources = set() if sources is None else sources
        self._transforms = transforms
        self._tracked_results: FunctionResults | None = None
        self["results"] = None

    def handle_event(self, event: Event) -> None:
        """Handle an event.

        Args:
            event: The event to handle.
        """
        if (
            event.event_type == EventType.FINISHED_EVALUATION
            and "results" in event.data
            and event.source in self._sources
        ):
            results = event.data["results"]
            if self["results"] is None:
                self._tracked_results = None
            filtered_results: FunctionResults | None = None
            match self._what:
                case "best":
                    filtered_results = _update_optimal_result(
                        self._tracked_results,
                        results,
                        self._constraint_tolerance,
                    )
                case "last":
                    filtered_results = _get_last_result(
                        results,
                        self._constraint_tolerance,
                    )
            if filtered_results is not None:
                self._tracked_results = filtered_results
                if self._transforms is not None:
                    filtered_results = filtered_results.transform_from_optimizer(
                        self._transforms
                    )
                self["results"] = filtered_results
