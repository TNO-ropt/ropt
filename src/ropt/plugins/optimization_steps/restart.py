"""This module defines the protocol to be followed by restart steps."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Tuple

from ropt.config.plan import RestartStepConfig
from ropt.optimization import BasicStep, Plan, PlanContext

if TYPE_CHECKING:
    from ropt.results import Results


class DefaultRestartStep(BasicStep):
    """The default restart plan."""

    def __init__(
        self,
        config: Dict[str, Any],
        context: PlanContext,  # noqa: ARG002
        plan: Plan,
    ) -> None:
        """Initialize a restart step.

        Args:
            config:  Optimizer configuration
            context: Context in which the plan runs
            plan:    The plan containing this step
        """
        self._config = RestartStepConfig.model_validate(config)
        self._plan = plan
        self._restart_idx = 0

    def run(self) -> bool:
        """Run the step.

        Returns:
            True if a user abort occurred.
        """
        if self._restart_idx < self._config.max_restarts:
            self._plan.restart(self._config.label)
        self._restart_idx += 1
        return False

    def set_metadata(self, results: Tuple[Results, ...]) -> Tuple[Results, ...]:
        """Update the results with metadata.

        Args:
            results: The results to track
        """
        if (
            self._restart_idx <= self._config.max_restarts
            and self._config.metadata_key is not None
        ):
            for item in results:
                item.metadata[self._config.metadata_key] = self._restart_idx
        return results
