"""This module implements the default setvar step."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from pydantic import BaseModel, ConfigDict

from ropt.config.utils import StrOrTuple  # noqa: TCH001
from ropt.exceptions import PlanError
from ropt.plugins.plan.base import PlanStep

if TYPE_CHECKING:
    from ropt.config.plan import StepConfig
    from ropt.plan import Plan


class DefaultSetStepExprWith(BaseModel):
    """Parameters used by the default setvar step.

    Attributes:
        expr: Expression used to set the variable
    """

    expr: str

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        arbitrary_types_allowed=True,
    )


class DefaultSetStepVarValueWith(BaseModel):
    """Parameters used by the default setvar step.

    Attributes:
        var:   The variable to set
        value: The value
    """

    var: str
    value: Any
    keys: StrOrTuple = ()

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        arbitrary_types_allowed=True,
    )


class DefaultSetStep(PlanStep):
    """The default set step."""

    def __init__(self, config: StepConfig, plan: Plan) -> None:
        """Initialize a default setvar step.

        Args:
            config: The configuration of the step
            plan:   The plan that runs this step
        """
        super().__init__(config, plan)

        expr: Optional[str] = None
        value: Any

        self._keys: Tuple[str, ...] = ()
        if isinstance(config.with_, str):
            expr = config.with_
        elif "expr" in config.with_:
            expr = DefaultSetStepExprWith.model_validate(config.with_).expr
        elif "var" in config.with_:
            with_ = DefaultSetStepVarValueWith.model_validate(config.with_)
            var = with_.var.strip()
            value = with_.value
            self._keys = with_.keys
        else:
            msg = "Either `expr` or `var` must be provided"
            raise ValueError(msg)

        if expr is not None:
            var, sep, value = expr.partition("=")
            if sep != "=":
                msg = f"Invalid expression: {expr}"
                raise PlanError(msg)

        pattern = re.findall(r"([^\[]+)|\[(.*?)\]", var.strip())
        self._var, *keys = (x.strip() for group in pattern for x in group if x)
        self._keys = tuple(keys)

        if self._var not in self._plan:
            msg = f"Unknown variable name: {self._var}"
            raise PlanError(msg)

        self._value = value

    def run(self) -> None:
        """Run the setvar step."""
        if self._keys:
            msg = f"Not a valid dict-like variable: {self._var}"
            try:
                expr = f"{self._var}" + "".join(f"[{key}]" for key in self._keys[:-1])
                target: Dict[str, Any] = self._plan.eval(expr)
            except PlanError as exc:
                raise PlanError(msg) from exc
            if not isinstance(target, (Mapping, Sequence)):
                raise PlanError(msg)
            target[self._plan.eval(self._keys[-1])] = self._plan.eval(self._value)
        else:
            self._plan[self._var] = self._plan.eval(self._value)
