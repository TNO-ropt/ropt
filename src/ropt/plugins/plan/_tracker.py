"""This module implements the default tracker handler."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ropt.enums import EventType
from ropt.plugins.plan.base import PlanHandler

from ._utils import _get_last_result, _update_optimal_result

if TYPE_CHECKING:
    import uuid

    from ropt.plan import Event, Plan
    from ropt.results import FunctionResults
    from ropt.transforms import OptModelTransforms


class DefaultTrackerHandler(PlanHandler):
    """The default handler for tracking optimization results.

    This handler listens for
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
    selected result, potentially transformed to the user domain if `transforms`
    were provided during initialization.
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

        This handler monitors [`Results`][ropt.results.Results] objects from
        specified `sources` (plan steps) and selects a single
        [`FunctionResults`][ropt.results.FunctionResults] object to retain based
        on the `what` criterion ('best' or 'last').

        The 'best' result is the one with the lowest weighted objective value
        encountered so far. The 'last' result is the most recently received
        valid result. Results can optionally be filtered by
        `constraint_tolerance` to ignore those violating constraints beyond the
        specified threshold.

        The `sources` parameter acts as a filter, determining which plan steps
        this handler should listen to. It should be a set containing the unique
        IDs (UUIDs) of the `PlanStep` instances whose results you want to track.
        When a [`FINISHED_EVALUATION`][ropt.enums.EventType.FINISHED_EVALUATION]
        event occurs, this handler checks if the ID of the step that emitted the
        event (`event.source`) is present in the `sources` set. If it is, the
        handler processes the results; otherwise, it ignores the event. If
        `sources` is `None` (the default), the handler will not track results
        from any source.

        Tracking logic (comparing 'best' or selecting 'last') operates on the
        results in the optimizer's domain. However, the final selected result
        that is made accessible via dictionary access (`handler["results"]`) is
        transformed to the user's domain if a `transforms` object is provided.

        Args:
            plan:                 The parent plan instance.
            what:                 Criterion for selecting results ('best' or 'last').
            constraint_tolerance: Optional threshold for filtering constraint violations.
            sources:              Optional set of step UUIDs whose results should be tracked.
            transforms:           Optional transforms object for user-domain results.
        """
        super().__init__(plan)
        self._what = what
        self._constraint_tolerance = constraint_tolerance
        self._sources = set() if sources is None else sources
        self._transforms = transforms
        self._tracked_results: FunctionResults | None = None
        self["results"] = None

    def handle_event(self, event: Event) -> None:
        """Handle incoming events from the plan.

        This method processes events emitted by the parent plan. It specifically
        listens for
        [`FINISHED_EVALUATION`][ropt.enums.EventType.FINISHED_EVALUATION] events
        originating from steps whose IDs are included in the `sources` set
        configured during initialization.

        If a relevant event containing results is received, this method updates
        the tracked result (`self["results"]`) based on the `what` criterion
        ('best' or 'last') and the optional `constraint_tolerance`.

        Args:
            event: The event object emitted by the plan.
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
