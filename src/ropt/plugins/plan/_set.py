"""This module implements the default setvar step."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Dict, List

from ropt.exceptions import PlanError
from ropt.plugins.plan.base import PlanStep

if TYPE_CHECKING:
    from ropt.config.plan import StepConfig
    from ropt.plan import Plan


class DefaultSetStep(PlanStep):
    """The default set step."""

    def __init__(self, config: StepConfig, plan: Plan) -> None:
        """Initialize a default set step.

        Args:
            config: The configuration of the step
            plan:   The plan that runs this step
        """
        super().__init__(config, plan)

        if not isinstance(config.with_, Mapping):
            msg = "`set` must be called with a var/value dict"
            raise PlanError(msg)

        self._vars: List[Any] = []
        self._values: List[Any] = []

        for var, value in config.with_.items():
            pattern = re.findall(r"([^\[]+)|\[(.*?)\]", var.strip())
            self._vars.append([x.strip() for group in pattern for x in group if x])
            self._values.append(value)
            if self._vars[-1][0] not in self._plan:
                msg = f"Unknown variable name: {self._vars[-1]}"
                raise PlanError(msg)

    def run(self) -> None:
        """Run the setvar step."""
        for var_and_keys, value in zip(self._vars, self._values):
            var, *keys = var_and_keys
            if not keys:
                self._plan[var] = self._plan.eval(value)
            else:
                msg = f"Not a valid dict-like variable: {var}"
                try:
                    expr = f"${var}" + "".join(f"[{key}]" for key in keys[:-1])
                    target: Dict[str, Any] = self._plan.eval(expr)
                except PlanError as exc:
                    raise PlanError(msg) from exc
                if not isinstance(target, (Mapping, Sequence)):
                    raise PlanError(msg)
                target[self._plan.eval(keys[-1])] = self._plan.eval(value)
