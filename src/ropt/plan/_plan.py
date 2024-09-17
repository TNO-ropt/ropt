"""This module defines the optimization plan object."""

from __future__ import annotations

import ast
import keyword
import re
from itertools import count
from numbers import Number
from typing import TYPE_CHECKING, Any, Dict, Final, List, Optional, Union

import numpy as np
from numpy.random import default_rng

from ropt.config.enopt.constants import DEFAULT_SEED
from ropt.exceptions import PlanError
from ropt.optimization import EventBroker
from ropt.plugins import PluginManager

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


MetaDataType = Dict[str, Union[int, float, bool, str]]


class OptimizerContext:
    """Context class for shared state across a plan.

    An optimizer context object holds the information and state shared across
    all steps in an optimization plan. This currently includes the following:

    - An [`Evaluator`][ropt.evaluator.Evaluator] callable for evaluating
      functions.
    - A seed for a random number generator used in stochastic gradient
      estimation.
    - An iterator that generates unique result IDs.
    - An event broker for connecting user-provided callbacks to optimization
      events.
    - A metadata dictionary that can be shared between steps.
    """

    def __init__(self, evaluator: Evaluator, seed: Optional[int] = None) -> None:
        """Initialize the optimization context.

        Args:
            evaluator:      The callable for running function evaluations
            seed:           Optional seed for the random number generator
        """
        self.evaluator = evaluator
        self.rng = default_rng(DEFAULT_SEED) if seed is None else default_rng(seed)
        self.result_id_iter = count()
        self.events = EventBroker()
        self.metadata: MetaDataType = {}


class Plan:
    """The plan class for executing optimization workflows."""

    def __init__(
        self,
        config: PlanConfig,
        optimizer_context: OptimizerContext,
        plugin_manager: Optional[PluginManager] = None,
    ) -> None:
        """Initialize a plan object.

        The plan requires a `PlanConfig` object and an `OptimizationContext`
        object. The `plugin_manager` argument is optional and allows you to
        specify plugins for the context and step objects that the plan may use.
        If not provided, only plugins installed via Python's standard entry
        points mechanism will be used.

        Args:
            config:            The optimizer configuration
            optimizer_context: The context in which the plan executes
            plugin_manager:    An optional plugin manager
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
        """Run the Plan.

        This method executes the steps of the plan. If a user abort event
        occurs, the method will return `True`.

        Returns:
            `True` if a user abort occurred; otherwise, `False`.
        """
        return self.run_steps(self._steps)

    def create_steps(self, step_configs: List[StepConfig]) -> List[PlanStep]:
        """Create step objects from step configs.

        Given a list of step configuration objects, this method returns a list
        of step objects, each configured according to its corresponding
        configuration.

        Args:
            step_configs: A list of step configuration objects.
        """
        return [
            self._plugin_manager.get_plugin("plan", method=step_config.run).create(
                step_config, self
            )
            for step_config in step_configs
        ]

    def run_steps(self, steps: List[PlanStep]) -> bool:
        """Execute a list of steps.

        This method executes a list of plan steps and returns `True` if the
        execution is aborted by the user.

        Args:
            steps: A list of steps to execute.

        Returns:
            `True` if the execution was aborted by the user; otherwise, `False`.
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

    def spawn(self, config: PlanConfig) -> Plan:
        """Spawn a new plan from the current plan.

        This method creates a new plan that shares the same optimization context
        and plugin manager as the current plan. However, it does not inherit
        other properties, such as variables.

        Args:
            config: The configuration of the new plan
        """
        return Plan(
            config,
            optimizer_context=self._optimizer_context,
            plugin_manager=self._plugin_manager,
        )

    def parse_value(self, value: Any) -> Any:  # noqa: ANN401
        """Parse a value as an expression or an interpolated string.

        If the value is not a string, it is returned unchanged. If it is a
        string, it is first stripped of leading and trailing whitespace and then
        parsed according to the following rules:

        - If the string starts with a `$`, it is evaluated using the
          [`eval`][ropt.plan.Plan.eval] method of the plan object. This will
          replace strings of the form `$identifier` with the corresponding plan
          value and evaluate strings of the form `${{ expr }}` as a mathematical
          expression, optionally containing variables in the form `$identifier`.
        - If the string does not start with a `$`, it is returned after
          interpolating any substrings delimited by `${{` and `}}` by passing
          them to the [`eval`][ropt.plan.Plan.eval] method.

        Note:
            The string `$$` is not interpolated but replaced with a single `$`.

        Args:
            value: The value to evaluate

        Returns:
            The result of the evaluation.
        """

        def _substitute(matched: re.Match[str]) -> str:
            value = matched.string[matched.start() : matched.end()]
            return "$" if value == "$$" else str(self.eval(value))

        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("$"):
                return self.eval(stripped)
            return re.sub(r"\${{(.*?)}}|\$\$|\$([^\W0-9][\w\.]*)", _substitute, value)
        return value

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

        If the value is a string, it is returned unchanged. Otherwise, it is
        evaluated as follows:

        - If the value is a string that starts with `$`, it is assumed to denote
          a plan variable, and its value is returned.
        - If the value is enclosed in `${{` and `}}` delimiters, these
          delimiters are removed. The string is then evaluated as a mathematical
          expression, with variables of the form `$identifier` replaced by their
          corresponding plan variable values.
        - Arbitrary strings that do not start with `$` are treated as if they
          are enclosed in `${{` and `}}` delimiters and evaluated in the same
          way.

        Note:
            The result of a mathematical expression is restricted to numerical
            values, lists, and numpy arrays. However, plan variables can contain
            values of any type, so expressions of the form `$identifier` may
            evaluate to a result of any type.

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

        This method calls the `reset` method of the context object identified by
        `obj_id`. The effect of this operation depends on the specific
        implementation of the context object.

        Args:
            obj_id: The ID of the context object to reset.
        """
        if obj_id in self._context:
            self._context[obj_id].reset()

    def has_context(self, obj_id: str) -> bool:
        """Check if a context object exists.

        Returns `True` if the plan contains a context object with the given ID;
        otherwise, returns `False`.

        Args:
            obj_id: The ID of the context object to check.

        Returns:
            `True` if the object exists; otherwise, `False`.
        """
        return obj_id in self._context

    def update_context(self, obj_id: str, value: Any) -> None:  # noqa: ANN401
        """Update a context object.

        This method calls the `update` method of the context object identified
        by `obj_id` with the given value. The effect of this operation depends
        on the specific implementation of the context object.

        Args:
            obj_id: The context object ID
            value:  The value to use for the update
        """
        if obj_id not in self._context:
            msg = f"not a valid context: `{obj_id}`"
            raise PlanError(msg)
        self._context[obj_id].update(value)

    def __getitem__(self, name: str) -> Any:  # noqa: ANN401
        """Get the value of a plan variable.

        This method implements the `[]` operator on the plan object to retrieve
        the value of a plan variable.

        Args:
            name: The name of the variable.

        Returns:
            The value of the variable.
        """
        if name in self._vars:
            return self._vars[name]
        msg = f"Unknown plan variable: `{name}`"
        raise PlanError(msg)

    def __setitem__(self, name: str, value: Any) -> None:  # noqa: ANN401
        """Set a plan variable to the given value.

        This method implements the `[]` operator on the plan object to set the
        value of a plan variable.

        Args:
            name:  The name of the variable.
            value: The value to assign.
        """
        if not name.isidentifier():
            msg = f"Not a valid variable name: `{name}`"
            raise PlanError(msg)
        self._vars[name] = value

    def __contains__(self, name: str) -> bool:
        """Check if a variable exists.

        This method implements the `in` operator on the plan object to determine
        if a plan variable exists.

        Args:
            name: The name of the variable.

        Returns:
            `True` if the variable exists; otherwise, `False`.
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
