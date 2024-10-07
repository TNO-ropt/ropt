"""This module implements the default optimizer step."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from ropt.exceptions import PlanError
from ropt.plugins.plan.base import PlanStep

if TYPE_CHECKING:
    from ropt.config.plan import StepConfig
    from ropt.plan import Plan


class DefaultResetStepWith(BaseModel):
    """Parameters used by the default reset step."""

    context: str

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
    )


class DefaultResetStep(PlanStep):
    """The default reset step."""

    def __init__(self, config: StepConfig, plan: Plan) -> None:
        """Initialize a default reset context step.

        Args:
            config: The configuration of the step
            plan:   The plan that runs this step
        """
        super().__init__(config, plan)

        self._context = (
            config.with_
            if isinstance(config.with_, str)
            else DefaultResetStepWith.model_validate(config.with_).context
        )

    def run(self) -> None:
        """Run the reset step."""
        if not self.plan.has_context(self._context):
            msg = f"Env object `{self._context}` does not exist."
            raise PlanError(msg, step_name=self.step_config.name)
        self.plan.reset_context(self._context)
