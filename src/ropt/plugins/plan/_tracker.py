"""This module implements the default tracker event handler."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, assert_never

from ropt.enums import EventType
from ropt.plugins.plan.base import EventHandler

from ._utils import _get_last_result, _update_optimal_result

if TYPE_CHECKING:
    from ropt.plan import Event
    from ropt.results import FunctionResults


class DefaultTrackerHandler(EventHandler):
    """The default event handler for tracking optimization results.

    This event handler listens for
    [`FINISHED_EVALUATION`][ropt.enums.EventType.FINISHED_EVALUATION] events
    emitted by specified steps. It processes the
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
        *,
        what: Literal["best", "last"] = "best",
        constraint_tolerance: float | None = None,
    ) -> None:
        """Initialize a default tracker event handler.

        This event handler monitors [`Results`][ropt.results.Results] objects
        from specified `steps and selects a single
        [`FunctionResults`][ropt.results.FunctionResults] object to retain based
        on the `what` criterion ('best' or 'last').

        The 'best' result is the one with the lowest weighted objective value
        encountered so far. The 'last' result is the most recently received
        valid result. Results can optionally be filtered by
        `constraint_tolerance` to ignore those violating constraints beyond the
        specified threshold.

        Tracking logic (comparing 'best' or selecting 'last') operates on the
        results in the optimizer's domain. However, the final selected result
        that is made accessible via dictionary access (`handler["results"]`) is
        transformed to the user's domain.

        Args:
            plan:                 The parent plan instance.
            what:                 Criterion for selecting results ('best' or 'last').
            constraint_tolerance: Optional threshold for filtering constraint violations.
        """
        super().__init__()
        self._what = what
        self._constraint_tolerance = constraint_tolerance
        self._tracked_results: FunctionResults | None = None
        self["results"] = None

    def handle_event(self, event: Event) -> None:
        """Handle incoming events from the plan.

        This method processes events emitted by the parent plan. It specifically
        listens for
        [`FINISHED_EVALUATION`][ropt.enums.EventType.FINISHED_EVALUATION] events
        originating from steps that are included in the connected steps.

        If a relevant event containing results is received, this method updates
        the tracked result (`self["results"]`) based on the `what` criterion
        ('best' or 'last') and the optional `constraint_tolerance`.

        Args:
            event: The event object emitted by the plan.
        """
        if (results := event.data.get("results")) is None:
            return
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
            case _ as unreachable:
                assert_never(unreachable)

        if filtered_results is not None:
            self._tracked_results = filtered_results
            transforms = event.data["transforms"]
            if transforms is not None:
                filtered_results = filtered_results.transform_from_optimizer(
                    event.data["config"], transforms
                )
            self["results"] = filtered_results

    @property
    def event_types(self) -> set[EventType]:
        """Return the event types that are handled.

        Returns:
            A set of event types that are handled.
        """
        return {EventType.FINISHED_EVALUATION}
