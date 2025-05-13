"""This module implements the default tracker event handler."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ropt.enums import EventType
from ropt.plugins.plan.base import EventHandler, PlanStep

from ._utils import _get_last_result, _update_optimal_result

if TYPE_CHECKING:
    from ropt.plan import Event, Plan
    from ropt.results import FunctionResults


class DefaultTrackerHandler(EventHandler):
    """The default event handler for tracking optimization results.

    This event handler listens for
    [`FINISHED_EVALUATION`][ropt.enums.EventType.FINISHED_EVALUATION] events
    emitted by specified `sources` (plan steps). It processes the
    [`Results`][ropt.results.Results] objects contained within these events and
    selects a single [`FunctionResults`][ropt.results.FunctionResults] object to
    retain based on defined criteria.

    The criteria for selection are:

    - **`what='best'` (default):** Tracks the result with the lowest weighted
      objective value encountered so far.
    - **`what='last'`:** Tracks the most recently received valid result.

    Optionally, results can be filtered based on constraint violations using the
    `constraint_tolerance` parameter. If provided, any result violating
    constraints beyond this tolerance is ignored.

    The selected result (in the optimizer domain) is stored internally. The
    result accessible via dictionary access (`handler["results"]`) is the
    selected result, potentially transformed to the user domain.
    """

    def __init__(
        self,
        plan: Plan,
        *,
        sources: set[PlanStep] | None = None,
        what: Literal["best", "last"] = "best",
        constraint_tolerance: float | None = None,
    ) -> None:
        """Initialize a default tracker event handler.

        This event handler monitors [`Results`][ropt.results.Results] objects
        from specified `sources` (plan steps) and selects a single
        [`FunctionResults`][ropt.results.FunctionResults] object to retain based
        on the `what` criterion ('best' or 'last').

        The 'best' result is the one with the lowest weighted objective value
        encountered so far. The 'last' result is the most recently received
        valid result. Results can optionally be filtered by
        `constraint_tolerance` to ignore those violating constraints beyond the
        specified threshold.

        The `sources` parameter acts as a filter, determining which plan steps
        this event handler should listen to. If `sources` is `None`, events from
        all sources will be processed.

        Tracking logic (comparing 'best' or selecting 'last') operates on the
        results in the optimizer's domain. However, the final selected result
        that is made accessible via dictionary access (`handler["results"]`) is
        transformed to the user's domain.

        Args:
            plan:                 The parent plan instance.
            what:                 Criterion for selecting results ('best' or 'last').
            constraint_tolerance: Optional threshold for filtering constraint violations.
            sources:              Optional set of steps whose results should be tracked.
        """
        super().__init__(plan, sources)
        self._what = what
        self._constraint_tolerance = constraint_tolerance
        self._tracked_results: FunctionResults | None = None
        self["results"] = None

    def handle_event(self, event: Event) -> None:
        """Handle incoming events from the plan.

        This method processes events emitted by the parent plan. It specifically
        listens for
        [`FINISHED_EVALUATION`][ropt.enums.EventType.FINISHED_EVALUATION] events
        originating from steps that are included in the `sources` set configured
        during initialization.

        If a relevant event containing results is received, this method updates
        the tracked result (`self["results"]`) based on the `what` criterion
        ('best' or 'last') and the optional `constraint_tolerance`.

        Args:
            event: The event object emitted by the plan.
        """
        if (
            event.event_type == EventType.FINISHED_EVALUATION
            and "results" in event.data
            and (self.source_ids is None or event.source in self.source_ids)
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
                if event.config.transforms is not None:
                    filtered_results = filtered_results.transform_from_optimizer(
                        event.config.transforms
                    )
                self["results"] = filtered_results
