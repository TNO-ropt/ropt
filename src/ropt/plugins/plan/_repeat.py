"""This module implements the default optimizer step."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from pydantic import BaseModel, ConfigDict

from ropt.config.plan import PlanStepConfig  # noqa: TCH001
from ropt.plugins.plan.base import PlanStep

if TYPE_CHECKING:
    from ropt.plan import Plan


class DefaultRepeatStep(PlanStep):
    """The default repeat step.

    This step executes a defined list of steps in sequence for a specified
    number of repetitions. Optionally, it can store the current iteration number
    in a variable for tracking purposes. Each iteration preserves the original
    order of steps, ensuring sequential execution rather than parallel.

    The repeat step uses the [`DefaultRepeatStepWith`]
    [ropt.plugins.plan._repeat.DefaultRepeatStep.DefaultRepeatStepWith]
    configuration class to interpret the `with` field of the
    [`PlanStepConfig`][ropt.config.plan.PlanStepConfig] in a plan configuration.
    """

    class DefaultRepeatStepWith(BaseModel):
        """Parameters used by the default repeat step.

        The `steps` parameter defines a sequence of actions that are executed
        repeatedly, with the number of repetitions determined by `iterations`.
        If `var` is specified, it will be set to the current iteration count
        before each round of steps, allowing for iteration-specific actions or
        tracking.

        Attributes:
            iterations: The number of repetitions to perform.
            steps:      The list of steps to repeat in each iteration.
            var:        Optional variable to update with the current iteration count.
        """

        iterations: int
        steps: List[PlanStepConfig]
        var: Optional[str] = None

        model_config = ConfigDict(
            extra="forbid",
            validate_default=True,
        )

    def __init__(self, config: PlanStepConfig, plan: Plan) -> None:
        """Initialize a default optimizer step.

        Args:
            config: The configuration of the step.
            plan:   The plan that runs this step.
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
