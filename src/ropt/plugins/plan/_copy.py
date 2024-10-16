"""This module implements the default setvar step."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import TYPE_CHECKING

from ropt.exceptions import PlanError
from ropt.plugins.plan.base import PlanStep

if TYPE_CHECKING:
    from ropt.config.plan import StepConfig
    from ropt.plan import Plan


class DefaultCopyStep(PlanStep):
    """The default copy step."""

    def __init__(self, config: StepConfig, plan: Plan) -> None:
        """Initialize a default copy step.

        Args:
            config: The configuration of the step
            plan:   The plan that runs this step
        """
        super().__init__(config, plan)

        if not isinstance(config.with_, Mapping):
            msg = "`set` must be called with a var/value dict"
            raise PlanError(msg)
        self._with = config.with_

    def run(self) -> None:
        """Run the copy step."""
        for to_, from_ in self._with.items():
            self._plan[to_] = deepcopy(self._plan.eval(from_))
