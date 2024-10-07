"""This module implements the default setvar step."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from pydantic import BaseModel, ConfigDict

from ropt.plugins.plan.base import PlanStep

if TYPE_CHECKING:
    from ropt.config.plan import StepConfig
    from ropt.plan import Plan


class DefaultMetadataStepWith(BaseModel):
    """Parameters used by the default setvar step.

    Attributes:
        metadata: Data to set into the metadata
    """

    metadata: Dict[str, Any]

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        arbitrary_types_allowed=True,
    )


class DefaultMetadataStep(PlanStep):
    """The default metadata step."""

    def __init__(self, config: StepConfig, plan: Plan) -> None:
        """Initialize a default metadata step.

        Args:
            config: The configuration of the step
            plan:   The plan that runs this step
        """
        super().__init__(config, plan)

        self._with = DefaultMetadataStepWith.model_validate(
            config.with_ if "metadata" in config.with_ else {"metadata": config.with_}
        )

    def run(self) -> None:
        """Run the metadata step."""
        metadata = self._plan.optimizer_context.metadata
        for key, expr in self._with.metadata.items():
            if expr is None:
                del metadata[key]
            else:
                metadata[key] = self.plan.parse_value(expr)
