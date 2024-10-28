"""This module implements the default optimizer step."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Union

from pydantic import BaseModel, ConfigDict

from ropt.config.plan import RunStepConfig, SetStepConfig  # noqa: TCH001
from ropt.plugins.plan.base import RunStep

if TYPE_CHECKING:
    from ropt.plan import Plan


class DefaultRepeatStep(RunStep):
    """The default optimizer step."""

    class DefaultRepeatStepWith(BaseModel):
        """Parameters used by the default repeat step.

        Attributes:
            iterations: The number of repetitions
            steps:      The steps to repeat
            var:        The variable to update with the counter value
        """

        iterations: int
        steps: List[Union[SetStepConfig, RunStepConfig]]
        var: Optional[str] = None

        model_config = ConfigDict(
            extra="forbid",
            validate_default=True,
        )

    def __init__(self, config: RunStepConfig, plan: Plan) -> None:
        """Initialize a default optimizer step.

        Args:
            config: The configuration of the step
            plan:   The plan that runs this step
        """
        super().__init__(config, plan)

        with_ = self.DefaultRepeatStepWith.model_validate(config.with_)
        self._num = with_.iterations
        self._steps = self.plan.create_steps(with_.steps)
        self._var = with_.var

    def run(self) -> None:
        """Run the steps repeatedly."""
        for idx in range(self._num):
            if self._var is not None:
                self.plan[self._var] = idx
            self.plan.run_steps(self._steps)
            if self.plan.aborted:
                break
