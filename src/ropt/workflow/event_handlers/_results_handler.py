"""This module implements the default result_handler event handler."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, assert_never

import numpy as np

from ropt.enums import EnOptEventType
from ropt.results import FunctionResults

from .base import EventHandler

if TYPE_CHECKING:
    from collections.abc import Callable

    from ropt.events import EnOptEvent
    from ropt.results import DomainType, Results


class ResultsHandler(EventHandler):
    """Track a single optimization result based on selection criteria.

    Listens for `FINISHED_EVALUATION` events and retains either the best
    (lowest weighted objective) or most recent valid result. Optionally
    filters by constraint tolerance.

    See [Optimization Workflows](../usage/workflows.md#result_handler) for full
    details on selection criteria and domain handling.
    """

    def __init__(
        self,
        *,
        what: Literal["best", "last"] = "best",
        constraint_tolerance: float | None = None,
        domain: DomainType = "user",
        filter: Callable[[Results], bool] | None = None,  # noqa: A002
    ) -> None:
        """Initialize the ResultsHandler.

        Args:
            what:                 Criterion for selecting results ('best' or 'last').
            constraint_tolerance: Optional threshold for filtering constraint violations.
            domain:               Domain in which to store the results ('user' or 'optimizer').
            filter:               Optional callable to filter results based on custom logic.
        """
        super().__init__()
        self._what = what
        self._constraint_tolerance = constraint_tolerance
        self._domain = domain
        self._filter = filter
        self._best_results: FunctionResults | None = None
        self["results"] = None

    def handle_event(self, event: EnOptEvent) -> None:
        """Handle incoming events.

        Processes `FINISHED_EVALUATION` events and updates the tracked result
        based on the configured criterion and constraint tolerance.

        Args:
            event: The event object.
        """
        results: tuple[FunctionResults, ...] = tuple(
            item
            for item in event.results
            if isinstance(item, FunctionResults)
            and item.functions is not None
            and (self._filter(item) if self._filter else True)
            and not _violates_constraint(item, self._constraint_tolerance)
        )
        if not results:
            return

        if self["results"] is None:
            self._best_results = None

        def _get_target_objective(result: FunctionResults) -> float:
            assert result.functions is not None
            return result.functions.target_objective.item()

        def _transform(result: FunctionResults) -> FunctionResults:
            return (
                result.transform_from_optimizer(event.context)
                if self._domain == "user"
                else result
            )

        match self._what:
            case "best":
                if self._best_results is not None:
                    results = (self._best_results, *results)
                best = min(results, key=_get_target_objective)
                if best is not self._best_results:
                    self._best_results = best
                    self["results"] = _transform(best)
            case "last":
                self["results"] = _transform(results[-1])
            case _ as unreachable:
                assert_never(unreachable)

    @property
    def event_types(self) -> set[EnOptEventType]:
        """The event types that are handled.

        Returns:
            A set of event types that are handled.
        """
        return {EnOptEventType.FINISHED_EVALUATION}


def _violates_constraint(results: Results, tolerance: float | None) -> bool:
    if tolerance is None:
        return False

    assert isinstance(results, FunctionResults)
    if results.constraint_info is None:
        return False

    for violations in (
        results.constraint_info.bound_violation,
        results.constraint_info.linear_violation,
        results.constraint_info.nonlinear_violation,
    ):
        if violations is not None and np.any(violations > tolerance):
            return True

    return False
