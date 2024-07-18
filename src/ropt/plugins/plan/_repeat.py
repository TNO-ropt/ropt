"""This module implements the default optimizer step."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from pydantic import BaseModel, ConfigDict

from ropt.config.plan import StepConfig  # noqa: TCH001
from ropt.plugins.plan.base import PlanStep

if TYPE_CHECKING:
    from ropt.plan import Plan


class DefaultRepeatStepWith(BaseModel):
    """Parameters used by the default repeat step.

    Attributes:
        iterations:  The number of repetitions
        steps:       The steps to repeat
        counter_var: The variable to update with the counter value
    """

    iterations: int
    steps: List[StepConfig]
    counter_var: Optional[str] = None

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
    )


class DefaultRepeatStep(PlanStep):
    """The default optimizer step."""

    def __init__(self, config: StepConfig, plan: Plan) -> None:
        """Initialize a default optimizer step.

        Args:
            config: The configuration of the step
            plan:   The plan that runs this step
        """
        super().__init__(config, plan)

        with_ = DefaultRepeatStepWith.model_validate(config.with_)
        self._num = with_.iterations
        self._steps = self.plan.create_steps(with_.steps)
        self._counter_var = with_.counter_var

    def run(self) -> bool:
        """Run the steps repeatedly.

        Returns:
            True if a user abort occurred.
        """
        for idx in range(self._num):
            if self._counter_var is not None:
                self.plan[self._counter_var] = idx
            if self.plan.run_steps(self._steps):
                return True
        return False
