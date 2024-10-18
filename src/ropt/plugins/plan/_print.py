"""This module implements the default print step."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from ropt.plugins.plan.base import PlanStep

if TYPE_CHECKING:
    from ropt.config.plan import StepConfig
    from ropt.plan import Plan


class DefaultPrintWith(BaseModel):
    """Parameters used by the default metadata results handler.

    Attributes:
        data: Data to set into the metadata
        tags: Tags of the sources to track
    """

    message: str

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        arbitrary_types_allowed=True,
    )


class DefaultPrintStep(PlanStep):
    """The default print step."""

    def __init__(self, config: StepConfig, plan: Plan) -> None:
        """Initialize a default print step.

        Args:
            config: The configuration of the step
            plan:   The plan that runs this step
        """
        super().__init__(config, plan)
        self._with = (
            DefaultPrintWith.model_validate({"message": config.with_})
            if isinstance(config.with_, str)
            else DefaultPrintWith.model_validate(config.with_)
        )

    def run(self) -> None:
        """Run the print step."""
        print(self.plan.interpolate_string(self._with.message))  # noqa: T201
