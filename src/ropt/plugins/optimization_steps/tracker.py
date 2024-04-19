"""This module defines the protocol to be followed by optimization steps."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Set, Tuple, Union

from ropt.config.plan import TrackerStepConfig
from ropt.exceptions import ConfigError
from ropt.optimization import TrackerStep

from ._utils import _get_last_result, _update_optimal_result

if TYPE_CHECKING:
    from ropt.optimization import Plan, PlanContext
    from ropt.results import FunctionResults, Results


class DefaultTrackerStep(TrackerStep):
    """The default evaluator step."""

    def __init__(
        self,
        config: Dict[str, Any],
        context: PlanContext,
        plan: Plan,  # noqa: ARG002
    ) -> None:
        """Initialize a default tracking step.

        Args:
            config:  The configuration of the step
            context: Context in which the step runs
            plan:    The plan containing this step
        """
        self._config = TrackerStepConfig.model_validate(config)
        self._context = context

        if self._config.id is not None:
            if self._config.id in self._context.results:
                msg = f"Duplicate step ID: {self._config.id}"
                raise ConfigError(msg)
            self._context.results[self._config.id] = None

        self._results: Optional[FunctionResults] = None
        self._track = (
            {self._config.source}
            if isinstance(self._config.source, str)
            else self._config.source
        )

    @property
    def id(self) -> Optional[str]:
        """The ID of the tracker.

        Returns:
            The tracker ID.
        """
        return self._config.id

    def reset(self) -> None:
        """Reset the results."""
        self._results = None

    def track_results(
        self, results: Tuple[Results, ...], tracker_id: Union[str, Set[str]]
    ) -> None:
        """Track results.

        Args:
            results:    The results to track
            tracker_id: ID of the step producing the results
        """
        for track in self._track:
            if track == tracker_id:
                if self._config.type_ == "optimal_result":
                    updated_optimal_result = _update_optimal_result(
                        self._results,
                        results,
                        self._config.constraint_tolerance,
                    )
                    if updated_optimal_result is not None:
                        self._results = updated_optimal_result
                        self._context.results[self._config.id] = updated_optimal_result
                elif self._config.type_ == "last_result":
                    last_result = _get_last_result(
                        results, self._config.constraint_tolerance
                    )
                    if last_result is not None:
                        self._context.results[self._config.id] = last_result
