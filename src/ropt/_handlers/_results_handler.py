"""This module implements the best-result event handler."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, assert_never, override

import numpy as np

from ropt._logging import get_logger
from ropt.enums import EnOptEventType
from ropt.results import FunctionResults, GradientResults

from .base import EventHandler

if TYPE_CHECKING:
    from collections.abc import Callable

    from ropt.events import EnOptEvent
    from ropt.results import Results

_logger = get_logger(__name__)


class ResultsHandler(EventHandler):
    """Track a single optimization result based on selection criteria.

    Listens for `FINISHED_EVALUATION` events and retains either the best
    (lowest weighted objective) or most recent valid result. Optionally
    filters by constraint tolerance. The selected result is accessible via the
    [`result`][ropt.ResultsHandler.result] property or
    `handler["results"]`.

    The gradient computed at the selected result, if there is one, is available
    as `handler["gradient"]`. It may arrive in a later event than the result it
    belongs to, and stays `None` for a point where no gradient was computed.

    See [Result Handlers](../results/handlers.md#resultshandler) for full
    details on selection criteria and scaling.
    """

    def __init__(
        self,
        *,
        what: Literal["best", "last"] = "best",
        constraint_tolerance: float | None = None,
        filter: Callable[[Results], bool] | None = None,  # ruff: ignore[builtin-argument-shadowing]
    ) -> None:
        """Initialize the ResultsHandler.

        Constraint violations are compared in the domain the optimizer works in,
        so a scale applies to them as well.

        Args:
            what:                 Criterion for selecting results ('best' or 'last').
            constraint_tolerance: Optional threshold for constraint violations.
            filter:               Optional callable to filter results based on custom logic.
        """
        super().__init__()
        self._what = what
        self._constraint_tolerance = constraint_tolerance
        self._filter = filter
        self._best_results: FunctionResults | None = None
        self["results"] = None
        self["gradient"] = None

    @property
    def result(self) -> FunctionResults | None:
        """The selected (best or last) result, or `None` if none is available."""
        selected: FunctionResults | None = self["results"]
        return selected

    @property
    def gradient(self) -> GradientResults | None:
        """The gradient at the selected result, or `None` if there is none."""
        selected: GradientResults | None = self["gradient"]
        return selected

    @override
    def _handle_event(self, event: EnOptEvent) -> None:
        self._select(event)
        self._attach_gradient(event)

    def _select(self, event: EnOptEvent) -> None:
        results: tuple[FunctionResults, ...] = tuple(
            item
            for item in event.results
            if isinstance(item, FunctionResults)
            and item.target_objective is not None
            # With realization_min_success = 0, a failed evaluation has a NaN target.
            and not np.isnan(item.target_objective)
            and (self._filter(item) if self._filter else True)
            and not _violates_constraint(item, self._constraint_tolerance)
        )
        if not results:
            return

        # Clearing the stored result is how a consumer restarts the tracking,
        # so the best seen so far must go with it.
        if self["results"] is None:
            self._best_results = None
            self["gradient"] = None

        def _get_target_objective(result: FunctionResults) -> float:
            assert result.target_objective is not None
            return result.target_objective.item()

        match self._what:
            case "best":
                if self._best_results is not None:
                    results = (self._best_results, *results)
                best = min(results, key=_get_target_objective)
                if best is not self._best_results:
                    self._best_results = best
                    _logger.info("New best objective: %g", _get_target_objective(best))
                    self._store(best)
            case "last":
                self._store(results[-1])
            case _ as unreachable:
                assert_never(unreachable)

    def _store(self, results: FunctionResults) -> None:
        self["results"] = results
        self["gradient"] = None

    def _attach_gradient(self, event: EnOptEvent) -> None:
        # A gradient may reach the handler after the result it belongs to, so
        # every event is searched, not only the one that selected the result.
        selected: FunctionResults | None = self["results"]
        if selected is None:
            return
        for item in event.results:
            if (
                isinstance(item, GradientResults)
                and item.uses(selected)
                and _at_same_point(item, selected)
            ):
                self["gradient"] = item
                return

    @property
    @override
    def event_types(self) -> set[EnOptEventType]:
        """The event types that are handled.

        Returns:
            A set of event types that are handled.
        """
        return {EnOptEventType.FINISHED_EVALUATION}


def _at_same_point(gradient: GradientResults, function: FunctionResults) -> bool:
    # The test the evaluator applies when it reuses a function result for a gradient.
    return np.allclose(
        gradient.scaled.variables, function.scaled.variables, rtol=0.0, atol=1e-15
    )


def _violates_constraint(results: Results, tolerance: float | None) -> bool:
    if tolerance is None:
        return False

    assert isinstance(results, FunctionResults)
    if results.scaled.constraint_info is None:
        return False

    for violations in (
        results.scaled.constraint_info.bound_violation,
        results.scaled.constraint_info.linear_violation,
        results.scaled.constraint_info.nonlinear_violation,
    ):
        if violations is not None and np.any(violations > tolerance):
            return True

    return False
