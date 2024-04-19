"""This module defines the protocol to be followed by optimization steps."""

from __future__ import annotations

from typing import Any, Dict

from ropt.config.plan import UpdateConfigStepConfig
from ropt.optimization import BasicStep, Plan, PlanContext
from ropt.utils import update_dict


class DefaultUpdateConfigStep(BasicStep):
    """The default variable configuration step."""

    def __init__(
        self, config: Dict[str, Any], context: PlanContext, plan: Plan
    ) -> None:
        """Initialize a variables step.

        Args:
            config:  The configuration of the step
            context: Context in which the plan runs
            plan:    The plan containing this step
        """
        self._config = UpdateConfigStepConfig.model_validate(config)
        self._context = context
        self._plan = plan

    def run(self) -> bool:
        """Run the step.

        Returns:
            True if a user abort occurred.
        """
        updates = self._config.updates
        initial_variables = self._config.initial_variables
        if initial_variables is not None:
            if initial_variables not in self._context.results:
                msg = f"No result available with name `{initial_variables}`"
                raise RuntimeError(msg)
            result = self._context.results.get(initial_variables)
            if result is not None and result.functions is not None:
                unscaled_variables = result.evaluations.unscaled_variables
                if unscaled_variables is None:
                    unscaled_variables = result.evaluations.variables
                updates = update_dict(
                    updates, {"variables": {"initial_values": unscaled_variables}}
                )
        if updates:
            self._plan.update_enopt_config(updates)
        return False
