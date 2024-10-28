"""This module implements the default print step."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from ropt.plugins.plan.base import RunStep

if TYPE_CHECKING:
    from ropt.config.plan import RunStepConfig
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


class DefaultPrintStep(RunStep):
    """The default print step."""

    def __init__(self, config: RunStepConfig, plan: Plan) -> None:
        """Initialize a default print step.

        Args:
            config: The configuration of the step
            plan:   The plan that runs this step
        """
        super().__init__(config, plan)
        _with = (
            DefaultPrintWith.model_validate({"message": config.with_})
            if isinstance(config.with_, str)
            else DefaultPrintWith.model_validate(config.with_)
        )
        self._message = _with.message.strip()
        if not (self._message.startswith("$[[") and self._message.endswith("]]")):
            self._message = "$[[" + self._message + "]]"

    def run(self) -> None:
        """Run the print step."""
        print(self.plan.eval(self._message))  # noqa: T201
