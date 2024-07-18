"""This module implements the default optimizer step."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from pydantic import BaseModel, ConfigDict

from ropt.exceptions import PlanError
from ropt.plan import ContextUpdateDict
from ropt.plugins.plan.base import PlanStep

if TYPE_CHECKING:
    from ropt.config.plan import StepConfig
    from ropt.plan import Plan


class DefaultUpdateStepWith(BaseModel):
    """Parameters used by the default update step."""

    context: str
    value: Dict[str, Any]

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
    )


class DefaultUpdateStep(PlanStep):
    """The default update context step."""

    def __init__(self, config: StepConfig, plan: Plan) -> None:
        """Initialize a default update step.

        Args:
            config: The configuration of the step
            plan:   The plan that runs this step
        """
        super().__init__(config, plan)

        self._with = DefaultUpdateStepWith.model_validate(config.with_)

    def run(self) -> bool:
        """Run the update step.

        Returns:
            True if a user abort occurred, always `False`.
        """
        if not self.plan.has_context(self._with.context):
            msg = f"Env object `{self._with.context}` does not exist."
            raise PlanError(msg, step_name=self.step_config.name)
        self.plan.update_context(
            self._with.context,
            ContextUpdateDict(step_name=self.step_config.name, data=self._with.value),
        )
        return False
