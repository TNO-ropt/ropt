"""This module defines the optimization plan object."""

from __future__ import annotations

import ast
import keyword
import re
from itertools import count
from numbers import Number
from typing import TYPE_CHECKING, Any, Dict, Final, List, Optional

import numpy as np
from numpy.random import default_rng

from ropt.config.enopt.constants import DEFAULT_SEED
from ropt.exceptions import PlanError
from ropt.plugins import PluginManager

from ._events import OptimizationEventBroker

if TYPE_CHECKING:
    from ropt.config.plan import PlanConfig, StepConfig
    from ropt.evaluator import Evaluator
    from ropt.plugins.plan.base import PlanStep

_VALID_TYPES: Final = (int, float, bool)
_VALID_RESULTS: Final = (list, np.ndarray, *_VALID_TYPES)
_UNARY_OPS: Final = (ast.UAdd, ast.USub, ast.Not)
_BIN_OPS: Final = (ast.Add, ast.Sub, ast.Div, ast.FloorDiv, ast.Mult, ast.Mod, ast.Pow)
_BOOL_OPS: Final = (ast.Or, ast.And)
_CMP_OPS: Final = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)


class OptimizerContext:
    """Store the context in which an optimizer runs."""

    def __init__(self, evaluator: Evaluator, seed: Optional[int] = None) -> None:
        """Initialize the optimiation context.

        Args:
            evaluator:      The callable for running function evaluations
            seed:           Optional seed for the random number generator
        """
        self.evaluator = evaluator
        self.rng = default_rng(DEFAULT_SEED) if seed is None else default_rng(seed)
        self.result_id_iter = count()
        self.events = OptimizationEventBroker()


class Plan:
    """The plan object."""

    def __init__(
        self,
        config: PlanConfig,
        optimizer_context: OptimizerContext,
        plugin_manager: Optional[PluginManager] = None,
    ) -> None:
        """Initialize a plan object.

        Args:
            config:            Optimizer configuration
            optimizer_context: Context in which the plan executes
            plugin_manager:    Optional plugin manager
        """
        self._plan_config = config
        self._optimizer_context = optimizer_context
        self._vars: Dict[str, Any] = {}

        self._plugin_manager = (
            PluginManager() if plugin_manager is None else plugin_manager
        )
        self._context = {
            config.id: self._plugin_manager.get_plugin(
                "plan", method=config.init
            ).create(config, self)
            for config in config.context
        }
        self._steps = self.create_steps(config.steps)

    def run(self) -> bool:
        """Run the plan.

        Returns:
            Whether a user abort occurred.
        """
        return self.run_steps(self._steps)

    def create_steps(self, step_configs: List[StepConfig]) -> List[PlanStep]:
        """Create step objects from step configs.

        Args:
            step_configs: The configurations of the steps.
        """
        return [
            self._plugin_manager.get_plugin("plan", method=step_config.run).create(
                step_config, self
            )
            for step_config in step_configs
        ]

    def run_steps(self, steps: List[PlanStep]) -> bool:
        """Run the given steps.

        Args:
            steps: The steps to run

        Returns:
            Whether a user abort occurred.
        """
        return any(
            task.run() for task in steps if self._check_condition(task.step_config)
        )

    @property
    def plugin_manager(self) -> PluginManager:
        """Return the plugin manager used by the plan.

        Returns:
            The plugin manager.
        """
        return self._plugin_manager

    @property
    def optimizer_context(self) -> OptimizerContext:
        """Return the optimizer context of the plan.

        Returns:
            The optimizer context object used by the plan.
        """
        return self._optimizer_context

    def parse_value(self, value: Any) -> Any:  # noqa: ANN401
        """Parse a value as an expression or an interpolated string.

        If the value is a string, and starts with `$`, it is assumed to be an
        expression and it is evaluated. If it does not start with `$`, any
        embedded string starting with `$` sign are evaluated and interpolated
        into the string.

        If the value is not a string, it is passed through unchanged.

        Args:
            value: The value to evaluate

        Returns:
            The result of the evaluation.
        """

        def _substitute(matched: re.Match[str]) -> str:
            value = matched.string[matched.start() : matched.end()]
            return "$" if value == "$$" else str(self.parse_value(value))

        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("$"):
                return self.eval(stripped)
            return re.sub(r"\${{(.*?)}}|\$\$|\$([^\W0-9][\w\.]*)", _substitute, value)
        return value

    def spawn(self, config: PlanConfig) -> Plan:
        """Spawn a child plan.

        Args:
            config: The configuration of the new plan.
        """
        return Plan(
            config,
            optimizer_context=self._optimizer_context,
            plugin_manager=self._plugin_manager,
        )

    def _eval_expr(self, expr: str) -> Any:  # noqa: ANN401
        # Check for identifiers that are not preceded by $:
        for word in re.findall(r"(?<!\$)\b\w+\b", expr):
            if word.isidentifier() and not keyword.iskeyword(word):
                msg = f"Syntax error in expression: {expr}"
                raise PlanError(msg)

        # Remove $ from identifiers, before sending the string to the parser:
        stripped = expr
        for word in re.findall(r"(?<=\$)\b\w+\b", expr):
            if word.isidentifier() and not keyword.iskeyword(word):
                stripped = stripped.replace(f"${word}", word)

        # Parse the string:
        try:
            tree = ast.parse(stripped, mode="eval")
        except SyntaxError as exc:
            msg = f"Syntax error in expression: {expr}"
            raise PlanError(msg) from exc

        # Replace identifiers with their value and evaluate:
        if _is_valid(tree.body):
            replacer = _ReplaceFields(self)
            tree = ast.fix_missing_locations(replacer.visit(tree))
            try:
                result = eval(  # noqa: S307
                    compile(tree, "", mode="eval"), {"__builtins__": {}}, replacer.vars
                )
            except TypeError as exc:
                msg = f"Type error in expression: {expr}"
                raise PlanError(msg) from exc
            assert result is None or isinstance(result, _VALID_RESULTS)
            return result

        msg = f"Invalid expression: {expr}"
        raise PlanError(msg)

    def eval(self, value: Any) -> Any:  # noqa: ANN401
        """Evaluate the provided value as an expression.

        The value is evaluated as follows:

        - If the value is not a string, return unchanged
        - If the value is a string enclosed in a `${{ }}` pair the contents are
          evaluated recursively.
        - If value is a string that starts with `$` and is an identifier it is
          assumed to be a plan variable and its value is returned. The
          resulting value can have any type.
        - Otherwise the string is evaluated and the result returned. The type of
          the result is restricted to numerical values, lists and numpy array.

        Args:
            value: The expression to evaluate.

        Returns:
            The result of the expression.
        """
        if not isinstance(value, str):
            return value

        value = value.strip()

        # Recursively evaluate when enclosed in `${{ }}`:
        if value.startswith("${{") and value.endswith("}}"):
            return self.eval(value[3:-2].strip())

        # Identifiers are not evaluated, their value is returned unchanged:
        if value.startswith("$") and value[1:].isidentifier():
            return self[value[1:]]

        # Evaluate as an expression:
        return self._eval_expr(value)

    def _check_condition(self, config: StepConfig) -> bool:
        if config.if_ is not None:
            return bool(self.eval(config.if_))
        return True

    def reset_context(self, obj_id: str) -> None:
        """Reset the given context object.

        Args:
            obj_id: The ID of the object.
        """
        if obj_id in self._context:
            self._context[obj_id].reset()

    def has_context(self, obj_id: str) -> bool:
        """Check if a variable of field exists.

        Args:
            obj_id: name of the context object

        Returns:
            Whether the object exists.
        """
        return obj_id in self._context

    def update_context(self, name: str, value: Any) -> None:  # noqa: ANN401
        """Update a context object.

        Args:
            name:  The context
            value: The value to use for the update
        """
        if name not in self._context:
            msg = f"not a valid context: `{name}`"
            raise PlanError(msg)
        self._context[name].update(value)

    def __getitem__(self, name: str) -> Any:  # noqa: ANN401
        """Get the value of a variable.

        Args:
            name: the variable name

        Returns:
            The value of the variable.
        """
        if name in self._vars:
            return self._vars[name]
        msg = f"Unknown plan variable: `{name}`"
        raise PlanError(msg)

    def __setitem__(self, name: str, value: Any) -> None:  # noqa: ANN401
        """Set a variable .

        Args:
            name:  The variable
            value: The value to assign
        """
        if not name.isidentifier():
            msg = f"Not a valid variable name: `{name}`"
            raise PlanError(msg)
        self._vars[name] = value

    def __contains__(self, name: str) -> bool:
        """Check if a variable exists.

        Args:
            name: name of the variable

        Returns:
            Whether the variable exists.
        """
        return name in self._vars


def _is_valid(node: ast.AST) -> bool:  # noqa: PLR0911
    if isinstance(node, ast.Constant):
        return node.value is None or type(node.value) in _VALID_TYPES
    if isinstance(node, ast.UnaryOp):
        return isinstance(node.op, _UNARY_OPS) and _is_valid(node.operand)
    if isinstance(node, ast.BinOp):
        return (
            isinstance(node.op, _BIN_OPS)
            and _is_valid(node.left)
            and _is_valid(node.right)
        )
    if isinstance(node, ast.BoolOp):
        return (
            isinstance(node.op, _BOOL_OPS)
            and all(_is_valid(value) for value in node.values)  # noqa: PD011
        )
    if isinstance(node, ast.Compare):
        return all(isinstance(op, _CMP_OPS) for op in node.ops) and all(
            _is_valid(value) for value in node.comparators
        )
    if isinstance(node, ast.List):
        return all(_is_valid(item) for item in node.elts)
    return bool(isinstance(node, ast.Name))


class _ReplaceFields(ast.NodeTransformer):
    def __init__(self, plan: Plan) -> None:
        self._plan = plan
        self.vars: Dict[str, Any] = {}

    def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802
        value = self._plan[node.id]
        if value is None or isinstance(value, (Number, np.ndarray)):
            self.vars[node.id] = value
            return node
        msg = f"Error in expression: the type of `{node.id}` is not supported"
        raise PlanError(msg)
