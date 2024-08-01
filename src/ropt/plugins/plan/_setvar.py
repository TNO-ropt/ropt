"""This module implements the default setvar step."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, ConfigDict

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
        var: str
        value: Any

        if isinstance(config.with_, str):
            expr = config.with_
        elif "expr" in config.with_:
            expr = DefaultSetStepExprWith.model_validate(config.with_).expr
        elif "var" in config.with_:
            with_ = DefaultSetStepVarValueWith.model_validate(config.with_)
            var = with_.var
            value = with_.value
        else:
            msg = "Either `expr` or `var` must be provided"
            raise ValueError(msg)

        if expr is not None:
            var, sep, value = expr.partition("=")
            if sep != "=":
                msg = f"Invalid expression: {expr}"
                raise PlanError(msg, step_name=self._step_config.name)

        self._var = var.strip()
        if not self._var.isidentifier():
            msg = f"Invalid identifier: {self._var}"
            raise PlanError(msg, step_name=self._step_config.name)
        self._value = value

    def run(self) -> bool:
        """Run the setvar step.

        Returns:
            True if a user abort occurred, always `False`.
        """
        self._plan[self._var] = self._plan.eval(self._value)
        return False
