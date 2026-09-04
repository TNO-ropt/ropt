"""This module implements the default result_handler event handler."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, assert_never

import numpy as np

from ropt._logging import get_logger
from ropt.enums import EnOptEventType
from ropt.results import FunctionResults

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
    [`result`][ropt.components.event_handlers.ResultsHandler.result] property or
    `handler["results"]`.

    See [Result Handlers](../running/handlers.md#resultshandler) for full
    details on selection criteria and scaling.
    """

    def __init__(
        self,
        *,
        what: Literal["best", "last"] = "best",
        constraint_tolerance: float | None = None,
        scaled: bool = False,
        filter: Callable[[Results], bool] | None = None,  # ruff: ignore[builtin-argument-shadowing]
    ) -> None:
        """Initialize the ResultsHandler.

        Args:
            what:                 Criterion for selecting results ('best' or 'last').
            constraint_tolerance: Optional threshold for filtering constraint violations.
            scaled:               If `True`, store the value as the optimizer works
                                  with it: scaled and offset, with objectives and
                                  gradients negated where `maximize` is set. By
                                  default the value is unscaled first, restoring the
                                  quantities as configured.
            filter:               Optional callable to filter results based on custom logic.
        """
        super().__init__()
        self._what = what
        self._constraint_tolerance = constraint_tolerance
        self._scaled = scaled
        self._filter = filter
        self._best_results: FunctionResults | None = None
        self["results"] = None

    @property
    def result(self) -> FunctionResults | None:
        """The selected (best or last) result, or `None` if none is available."""
        selected: FunctionResults | None = self["results"]
        return selected

    def _handle_event(self, event: EnOptEvent) -> None:
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

        # Clearing the stored result is how a consumer restarts the tracking,
        # so the best seen so far must go with it.
        if self["results"] is None:
            self._best_results = None

        def _get_target_objective(result: FunctionResults) -> float:
            assert result.functions is not None
            return result.functions.target_objective.item()

        def _maybe_unscale(result: FunctionResults) -> FunctionResults:
            return result if self._scaled else result.unscale(event.context)

        match self._what:
            case "best":
                # The best so far competes with the new batch, so it is only
                # replaced by something better, and kept scaled to stay
                # comparable with what arrives next.
                if self._best_results is not None:
                    results = (self._best_results, *results)
                best = min(results, key=_get_target_objective)
                if best is not self._best_results:
                    self._best_results = best
                    _logger.info("New best objective: %g", _get_target_objective(best))
                    self["results"] = _maybe_unscale(best)
            case "last":
                self["results"] = _maybe_unscale(results[-1])
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
