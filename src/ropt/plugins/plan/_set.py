"""This module implements the default set step."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, MutableMapping, MutableSequence, Sequence
from typing import TYPE_CHECKING, Any, List, Union

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

        self._vars: List[Any] = []
        self._values: List[Any] = []

        for list_item in (
            [config.with_] if isinstance(config.with_, Mapping) else config.with_
        ):
            if not isinstance(list_item, Mapping):
                msg = "`set` must be called with a var/value dict or a list of var/value dicts"
                raise PlanError(msg)
            for var, value in list_item.items():
                pattern = re.findall(r"([^\[]+)|\[(.*?)\]", var.strip())
                self._vars.append([x.strip() for group in pattern for x in group if x])
                self._values.append(value)
                if self._vars[-1][0] not in self._plan:
                    msg = f"Unknown variable name: {self._vars[-1]}"
                    raise PlanError(msg)

    def run(self) -> None:
        """Run the set step."""
        for var_and_keys, value in zip(self._vars, self._values):
            var, *keys = var_and_keys
            if not keys:
                self._plan[var] = copy.deepcopy(self._plan.eval(value))
            else:
                msg = f"Not a valid dict-like variable: {var}"
                keys = [self.plan.eval("${{" + key + "}}") for key in keys]
                target: Union[MutableMapping[Any, Any], Sequence[Any]] = self._plan[var]
                try:
                    for key in keys[:-1]:
                        target = target[key]
                except KeyError:
                    raise PlanError(msg) from None
                if not isinstance(target, (MutableMapping, MutableSequence)):
                    raise PlanError(msg)
                target[keys[-1]] = copy.deepcopy(self._plan.eval(value))
