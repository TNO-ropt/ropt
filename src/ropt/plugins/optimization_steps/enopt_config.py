"""This module defines the protocol to be followed by optimization steps."""

from __future__ import annotations

from typing import Any, Dict

from ropt.config.enopt import EnOptConfig
from ropt.optimization import BasicStep, Plan, PlanContext


class DefaultEnOptConfigStep(BasicStep):
    """The default configuration step."""

    def __init__(
        self,
        enopt_config: Dict[str, Any],
        context: PlanContext,  # noqa: ARG002
        plan: Plan,
    ) -> None:
        """Initialize a default config step.

        Args:
            enopt_config: The configuration of the step
            context:      Context in which the plan runs
            plan:         The plan containing this step
        """
        self._enopt_config = EnOptConfig.model_validate(enopt_config)
        self._plan = plan

    def run(self) -> bool:
        """Run the step.

        Returns:
            True if a user abort occurred.
        """
        self._plan.set_enopt_config(self._enopt_config)
        return False
