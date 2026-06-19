"""This module implements the default tracker event handler."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, assert_never

from ropt.enums import EnOptEventType

from ._utils import _get_last_result, _update_optimal_result
from .base import EventHandler

if TYPE_CHECKING:
    from ropt.events import EnOptEvent
    from ropt.results import DomainType, FunctionResults


class Tracker(EventHandler):
    """Track a single optimization result based on selection criteria.

    Listens for `FINISHED_EVALUATION` events and retains either the best
    (lowest weighted objective) or most recent valid result. Optionally
    filters by constraint tolerance.

    See [Optimization Workflows](../usage/workflows.md#tracker) for full
    details on selection criteria and domain handling.
    """

    def __init__(
        self,
        *,
        what: Literal["best", "last"] = "best",
        constraint_tolerance: float | None = None,
        domain: DomainType = "user",
    ) -> None:
        """Initialize the Tracker.

        Args:
            what:                 Criterion for selecting results ('best' or 'last').
            constraint_tolerance: Optional threshold for filtering constraint violations.
            domain:               Domain in which to store the results ('user' or 'optimizer').
        """
        super().__init__()
        self._what = what
        self._constraint_tolerance = constraint_tolerance
        self._domain = domain
        self._tracked_results: FunctionResults | None = None
        self["results"] = None

    def handle_event(self, event: EnOptEvent) -> None:
        """Handle incoming events.

        Processes `FINISHED_EVALUATION` events and updates the tracked result
        based on the configured criterion and constraint tolerance.

        Args:
            event: The event object.
        """
        if not (results := event.results):
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
            if self._domain == "user":
                self["results"] = filtered_results.transform_from_optimizer(
                    event.context
                )
            else:
                self["results"] = filtered_results

    @property
    def event_types(self) -> set[EnOptEventType]:
        """The event types that are handled.

        Returns:
            A set of event types that are handled.
        """
        return {EnOptEventType.FINISHED_EVALUATION}
