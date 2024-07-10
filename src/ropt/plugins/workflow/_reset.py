"""This module implements the default optimizer step."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from ropt.exceptions import WorkflowError
from ropt.plugins.workflow.base import WorkflowStep

if TYPE_CHECKING:
    from ropt.config.workflow import StepConfig
    from ropt.workflow import Workflow


class DefaultResetStepWith(BaseModel):
    """Parameters used by the default reset step."""

    context: str

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
    )


class DefaultResetStep(WorkflowStep):
    """The default reset step."""

    def __init__(self, config: StepConfig, workflow: Workflow) -> None:
        """Initialize a default reset context step.

        Args:
            config:   The configuration of the step
            workflow: The workflow that runs this step
        """
        super().__init__(config, workflow)

        self._with = DefaultResetStepWith.model_validate(config.with_)

    def run(self) -> bool:
        """Run the reset step.

        Returns:
            True if a user abort occurred, always `False`.
        """
        if not self.workflow.has_context(self._with.context):
            msg = f"Env object `{self._with.context}` does not exist."
            raise WorkflowError(msg, step_name=self.step_config.name)
        self.workflow.reset_context(self._with.context)
        return False
