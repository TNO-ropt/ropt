"""This module implements the default set step."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, MutableMapping, MutableSequence
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union

from pydantic import ConfigDict, RootModel

from ropt.plugins.plan.base import PlanStep

if TYPE_CHECKING:
    from ropt.config.plan import PlanStepConfig
    from ropt.plan import Plan


class DefaultSetStep(PlanStep):
    """The default set step.

    Set steps are used to modify the contents of plan variables. They specify
    one or more variables to set, either as a dictionary of variable-name/value
    pairs, or as a list of such dictionaries.

    Variables containing dictionaries can be modified by specifying an entry
    using the `[]` operator. Similarly, variables containing objects with
    attributes can be modified using the `.` operator. These can be mixed and
    nested to any depth to modify complex variables. For example, the expression
    `$var['foo'].bar[0]` is valid if `var` contains a dict-like value with a
    `foo` entry that has a `bar` attribute containing a list.

    The evaluator step uses the [`DefaultSetStepWith`]
    [ropt.plugins.plan._set.DefaultSetStep.DefaultSetStepWith]
    configuration class to parse the `with` field of the
    [`PlanStepConfig`][ropt.config.plan.PlanStepConfig] used to specify this step
    in a plan configuration.

    Note: Dictionary vs Lists
        Multiple variables may be set in a single step, either by using a single
        dictionary of variable/value pairs or by providing a list of
        dictionaries. Both approaches are generally equivalent. However, if a
        dictionary is loaded from a JSON or YAML file, the order of the keys is
        not guaranteed. Keep this in mind if the variables are interdependent
        and the assignment order matters.
    """

    class DefaultSetStepWith(
        RootModel[Union[Dict[str, Any], Tuple[Dict[str, Any], ...]]]
    ):
        """Configuration for a set step.

        A set step is used to change the value of one or more plan variables.
        The parameters provided to the step should either be a dictionary of
        variable-name/value pairs, or a list of such dictionaries. When the set
        step is run by the plan, the variables are set to the given values.
        Variables that are referenced may be dictionaries or objects with
        attributes. These can be modified by using the `[]` or `.` operators
        (possibly nested) with the specified variable name.

        Info: Using Expressions in the Value
            Optionally, the supplied value may be an expression, following the rules
            of the [`eval`][ropt.plan.Plan.eval] method of the plan object executing
            the steps. Refer to the method's documentation for more information
            on supported expressions.

        Note: Dictionaries vs. Lists
            The parameters provided to the set step may be dictionaries or lists
            of dictionaries. Multiple variables can be set in this manner in
            both cases. However, care should be taken if a dictionary is used
            where the order of the keys is not well defined, for instance, if it
            has been read from a JSON file. If the order in which variables are
            set is important, it is recommended to use a list.
        """

    root: Union[Dict[str, Any], Tuple[Dict[str, Any], ...]]

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
    )

    def __init__(self, config: PlanStepConfig, plan: Plan) -> None:
        """Initialize a set step.

        Args:
            config: The configuration of the step.
            plan:   The plan that runs this step.
        """
        super().__init__(config, plan)
        expr = self.DefaultSetStepWith.model_validate(config.with_).root
        if isinstance(expr, Mapping):
            expr = (expr,)

        self._plan = plan
        self._targets: List[Any] = []
        self._names: List[str] = []
        self._keys: List[List[str]] = []
        self._values: List[Any] = []

        for list_item in expr:
            if not isinstance(list_item, Mapping):
                msg = "`set` must be called with a var/value dict or a list of var/value dicts"
                raise TypeError(msg)
            for target, value in list_item.items():
                pattern = re.findall(
                    r"([^\[^\.]+)|(\[.*?\])|(\.\b\w+\b)", target.strip()
                )
                name, *keys = [x.strip() for group in pattern for x in group if x]
                if name not in self._plan:
                    msg = f"Unknown variable name: {name}"
                    raise AttributeError(msg)
                self._targets.append(target)
                self._names.append(name)
                self._keys.append(
                    [
                        key[1:] if key.startswith(".") else "${{" + key[1:-1] + "}}"
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
                    raise AttributeError(msg) from None

    def _get_target(self, target: Any, key: str) -> Any:  # noqa: ANN401
        if key.startswith("${{"):
            if not isinstance(target, (MutableMapping, MutableSequence)):
                raise KeyError
            return target[self._plan.eval(key)]
        return getattr(target, key)

    def _set_target(self, target: Any, key: str, value: Any) -> None:  # noqa: ANN401
        if key.startswith("${{"):
            if not isinstance(target, (MutableMapping, MutableSequence)):
                raise KeyError
            target[self._plan.eval(key)] = copy.deepcopy(self._plan.eval(value))
        else:
            setattr(target, key, copy.deepcopy(self._plan.eval(value)))
