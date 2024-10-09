"""This module defines the optimization plan object."""

from __future__ import annotations

import ast
import keyword
import re
from itertools import count
from numbers import Number
from typing import TYPE_CHECKING, Any, Callable, Dict, Final, List, Optional, Tuple

import numpy as np
from numpy.random import default_rng

from ropt.config.enopt.constants import DEFAULT_SEED
from ropt.enums import EventType
from ropt.exceptions import PlanError
from ropt.optimization import Event
from ropt.plugins import PluginManager

if TYPE_CHECKING:
    from ropt.config.enopt import EnOptConfig
    from ropt.config.plan import PlanConfig, StepConfig
    from ropt.evaluator import Evaluator
    from ropt.plugins.plan.base import ContextObj, PlanStep

_VALID_TYPES: Final = (int, float, bool)
_VALID_RESULTS: Final = (list, np.ndarray, *_VALID_TYPES)
_UNARY_OPS: Final = (ast.UAdd, ast.USub, ast.Not)
_BIN_OPS: Final = (ast.Add, ast.Sub, ast.Div, ast.FloorDiv, ast.Mult, ast.Mod, ast.Pow)
_BOOL_OPS: Final = (ast.Or, ast.And)
_CMP_OPS: Final = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)


class OptimizerContext:
    """Context class for shared state across a plan.

    An optimizer context object holds the information and state shared across
    all steps in an optimization plan. This currently includes the following:

    - An [`Evaluator`][ropt.evaluator.Evaluator] callable for evaluating
      functions.
    - A seed for a random number generator used in stochastic gradient
      estimation.
    - An iterator that generates unique result IDs.
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
        self._subscribers: Dict[EventType, List[Callable[[Event], None]]] = {
            event: [] for event in EventType
        }

        self._plugin_manager = (
            PluginManager() if plugin_manager is None else plugin_manager
        )
        self._context: List[ContextObj] = [
            self._plugin_manager.get_plugin("plan", method=config.init).create(
                config, self
            )
            for config in config.context
        ]
        self._steps = self.create_steps(config.steps)
        self._aborted = False

    def run(self, *args: Any) -> Tuple[Any, ...]:  # noqa: ANN401
        """Run the Plan.

        This method executes the steps of the plan. If a user abort event
        occurs, the method will return `True`.
        """
        for var, value in self._plan_config.variables.items():
            self[var] = value
        len_args = len(args)
        len_inputs = len(self._plan_config.inputs)
        if len_args != len_inputs:
            msg = f"The number of inputs is incorrect: expected {len_inputs}, passed {len_args}"
            raise PlanError(msg)
        for name, arg in zip(self._plan_config.inputs, args):
            self[name] = arg
        self.run_steps(self._steps)
        missing = [name for name in self._plan_config.outputs if name not in self]
        if missing:
            msg = f"Missing outputs: {missing}"
            raise PlanError(msg)
        return tuple(self[name] for name in self._plan_config.outputs)

    def abort(self) -> None:
        """Abort the plan."""
        self._aborted = True

    @property
    def aborted(self) -> bool:
        """Whether the plan was aborted by the user.

        Returns:
            `True` if the user aborted the plan.
        """
        return self._aborted

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

    def run_steps(self, steps: List[PlanStep]) -> None:
        """Execute a list of steps.

        This method executes a list of plan steps and returns `True` if the
        execution is aborted by the user.

        Args:
            steps: A list of steps to execute.

        Returns:
            `True` if the execution was aborted by the user; otherwise, `False`.
        """
        for task in steps:
            if self._check_condition(task.step_config):
                task.run()
            if self._aborted:
                break

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

        In addition, any signals that are emitted by the spawned plan are
        forwarded to the observers that are connected to this plan. In other
        words, signals emitted by the spawn plan 'bubble' up to the current
        plan.

        Args:
            config: The configuration of the new plan
        """
        plan = Plan(
            config,
            optimizer_context=self._optimizer_context,
            plugin_manager=self._plugin_manager,
        )

        for event_type in EventType:
            for callback in self._subscribers[event_type]:
                plan.add_observer(event_type, callback)

        return plan

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

    def add_observer(
        self,
        event: EventType,
        callback: Callable[[Event], None],
    ) -> None:
        """Add an observer function.

        Observer functions will be called during optimization if an event of the
        given type occurs. The callable must accept an argument of the
        [`Event`][ropt.optimization.Event] class that contains information about
        the event that occurred.

        Args:
            event:    The type of events to react to
            callback: The function to call if the event is received
        """
        self._subscribers[event].append(callback)

    def emit_event(
        self,
        event_type: EventType,
        config: EnOptConfig,
        /,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Emit an event of the given type with given data.

        When called, an [`Event`][ropt.optimization.Event] object is constructed
        using the given `event_type` and `config` for its mandatory fields. When
        given, the additional keyword arguments are also passed to the
        [`Event`][ropt.optimization.Event] constructor to set the optional
        fields. All callbacks for the given event type, that were added by the
        `add_observer` method are then called using the newly constructed event
        object as their argument.

        Args:
            event_type: The type of event to emit
            config:     Optimization configuration used by the emitting object
            kwargs:     Keyword arguments used to create an optimization event
        """
        event = Event(event_type, config, **kwargs)
        for context in self._context:
            event = context.handle_event(event)
        for callback in self._subscribers[event.event_type]:
            callback(event)

    def _check_condition(self, config: StepConfig) -> bool:
        if config.if_ is not None:
            return bool(self.eval(config.if_))
        return True

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
