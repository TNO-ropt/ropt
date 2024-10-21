"""This module implements the default set step."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, MutableMapping, MutableSequence
from typing import TYPE_CHECKING, Any, List

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

        self._targets: List[Any] = []
        self._names: List[str] = []
        self._keys: List[List[str]] = []
        self._values: List[Any] = []

        for list_item in (
            [config.with_] if isinstance(config.with_, Mapping) else config.with_
        ):
            if not isinstance(list_item, Mapping):
                msg = "`set` must be called with a var/value dict or a list of var/value dicts"
                raise PlanError(msg)
            for target, value in list_item.items():
                pattern = re.findall(
                    r"([^\[^\.]+)|(\[.*?\])|(\.\b\w+\b)", target.strip()
                )
                name, *keys = [x.strip() for group in pattern for x in group if x]
                if name not in self._plan:
                    msg = f"Unknown variable name: {name}"
                    raise PlanError(msg)
                self._targets.append(target)
                self._names.append(name)
                self._keys.append(
                    [
                        key[1:] if key.startswith(".") else "{{" + key[1:-1] + "}}"
                        for key in keys
                    ]
                )
                self._values.append(value)

    def run(self) -> None:
        """Run the set step."""
        for target_string, name, keys, value in zip(
            self._targets, self._names, self._keys, self._values
        ):
            if not keys:
                self._plan[name] = copy.deepcopy(self._plan.eval(value))
            else:
                target = self._plan[name]
                try:
                    for key in keys[:-1]:
                        target = self._get_target(target, key)
                    self._set_target(target, keys[-1], value)
                except (AttributeError, KeyError):
                    msg = f"Invalid attribute access: {target_string}"
                    raise PlanError(msg) from None

    def _get_target(self, target: Any, key: str) -> Any:  # noqa: ANN401
        if key.startswith("{{"):
            if not isinstance(target, (MutableMapping, MutableSequence)):
                raise KeyError
            return target[self.plan.eval(key)]
        return getattr(target, key)

    def _set_target(self, target: Any, key: str, value: Any) -> None:  # noqa: ANN401
        if key.startswith("{{"):
            if not isinstance(target, (MutableMapping, MutableSequence)):
                raise KeyError
            target[self.plan.eval(key)] = copy.deepcopy(self._plan.eval(value))
        else:
            setattr(target, key, copy.deepcopy(self._plan.eval(value)))
