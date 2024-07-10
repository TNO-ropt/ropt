"""This module implements the default setvar step."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, ConfigDict

from ropt.exceptions import WorkflowError
from ropt.plugins.workflow.base import WorkflowStep

if TYPE_CHECKING:
    from ropt.config.workflow import StepConfig
    from ropt.workflow import Workflow


class DefaultSetStepWith(BaseModel):
    """Parameters used by the default setvar step.

    Attributes:
        var:   The variable to set
        value: The value
    """

    var: str
    value: Optional[Any] = None

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        arbitrary_types_allowed=True,
    )


class DefaultSetStep(WorkflowStep):
    """The default set step."""

    def __init__(self, config: StepConfig, workflow: Workflow) -> None:
        """Initialize a default setvar step.

        Args:
            config:   The configuration of the step
            workflow: The workflow that runs this step
        """
        self._value: Any
        super().__init__(config, workflow)
        if isinstance(config.with_, str):
            self._var, sep, self._value = config.with_.partition("=")
            if sep != "=":
                msg = f"Invalid expression: {config.with_}"
                raise WorkflowError(msg, step_name=self._step_config.name)
            self._var = self._var.strip()
            if not self._var.isidentifier():
                msg = f"Invalid identifier: {self._var}"
                raise WorkflowError(msg, step_name=self._step_config.name)
        else:
            with_ = DefaultSetStepWith.model_validate(config.with_)
            self._var = with_.var.strip()
            self._value = with_.value

    def run(self) -> bool:
        """Run the setvar step.

        Returns:
            True if a user abort occurred, always `False`.
        """
        self._workflow[self._var] = self._workflow.eval(self._value)
        return False
