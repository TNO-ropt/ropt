"""This module defines workflow object."""

from __future__ import annotations

import ast
import re
from itertools import count
from numbers import Number
from typing import TYPE_CHECKING, Any, Dict, Final, List, Optional

import numpy as np
from numpy.random import default_rng

from ropt.config.enopt.constants import DEFAULT_SEED
from ropt.exceptions import WorkflowError
from ropt.plugins import PluginManager

if TYPE_CHECKING:
    from ropt.config.workflow import StepConfig, WorkflowConfig
    from ropt.evaluator import Evaluator
    from ropt.plugins.workflow.base import WorkflowStep

_VALID_TYPES: Final = (int, float, bool)
_VALID_RESULTS: Final = (list, np.ndarray, *_VALID_TYPES)
_UNARY_OPS: Final = (ast.UAdd, ast.USub, ast.Not)
_BIN_OPS: Final = (ast.Add, ast.Sub, ast.Div, ast.FloorDiv, ast.Mult, ast.Mod, ast.Pow)
_BOOL_OPS: Final = (ast.Or, ast.And)
_CMP_OPS: Final = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)


class OptimizerContext:
    """Store the context in which a workflow runs."""

    def __init__(self, evaluator: Evaluator, seed: Optional[int] = None) -> None:
        """Initialize the workflow context.

        Args:
            evaluator:      The callable for running function evaluations
            seed:           Optional seed for the random number generator
        """
        self.evaluator = evaluator
        self.rng = default_rng(DEFAULT_SEED) if seed is None else default_rng(seed)
        self.result_id_iter = count()


class Workflow:
    """The workflow object."""

    def __init__(
        self,
        config: WorkflowConfig,
        context: OptimizerContext,
        plugin_manager: Optional[PluginManager] = None,
    ) -> None:
        """Initialize a workflow object.

        Args:
            config:         Optimizer configuration
            context:        Context in which the workflow executes
            plugin_manager: Optional plugin manager
        """
        self._workflow_config = config
        self._workflow_context = context

        self._plugin_manager = (
            PluginManager() if plugin_manager is None else plugin_manager
        )
        self._context = {
            config.id: self._plugin_manager.get_plugin(
                "workflow", method=config.init
            ).create(config, self)
            for config in config.context
        }
        self._steps = self.create_steps(config.steps)
        self._vars: Dict[str, Any] = {}

    def run(self) -> bool:
        """Run the workflow.

        Returns:
            Whether a user abort occurred.
        """
        return self.run_steps(self._steps)

    def create_steps(self, step_configs: List[StepConfig]) -> List[WorkflowStep]:
        """Create step objects from step configs.

        Args:
            step_configs: The configurations of the steps.
        """
        return [
            self._plugin_manager.get_plugin("workflow", method=step_config.run).create(
                step_config, self
            )
            for step_config in step_configs
        ]

    def run_steps(self, steps: List[WorkflowStep]) -> bool:
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
        """Return the plugin manager used by the workflow.

        Returns:
            The plugin manager.
        """
        return self._plugin_manager

    @property
    def optimizer_context(self) -> OptimizerContext:
        """Return the optimizer context of the workflow.

        Returns:
            The optimizer context object used by the workflow.
        """
        return self._workflow_context

    def parse_value(self, value: Any) -> Any:  # noqa: ANN401
        """Parse a value as a variable, context field or expression.

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
            if stripped.startswith("${{") and stripped.endswith("}}"):
                return self._eval(stripped[3:-2].strip())
            if stripped.startswith("$"):
                return self.__getitem__(stripped[1:])
            return re.sub(r"\${{(.*?)}}|\$\$|\$([^\W0-9][\w\.]*)", _substitute, value)
        return value

    def _eval(self, expr: str) -> Any:  # noqa: ANN401
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as exc:
            msg = f"Syntax error in workflow expression: {expr}"
            raise WorkflowError(msg) from exc

        if _is_valid(tree.body):
            replacer = _ReplaceFields(self)
            tree = ast.fix_missing_locations(replacer.visit(tree))
            try:
                result = eval(  # noqa: S307
                    compile(tree, "", mode="eval"), {"__builtins__": {}}, replacer.vars
                )
            except TypeError as exc:
                msg = f"Type error in workflow expression: {expr}"
                raise WorkflowError(msg) from exc
            assert result is None or isinstance(result, _VALID_RESULTS)
            return result

        msg = f"Invalid workflow expression: {expr}"
        raise WorkflowError(msg)

    def _check_condition(self, config: StepConfig) -> bool:
        if config.if_ is not None:
            if config.if_.strip().startswith("$"):
                return bool(self.parse_value(config.if_))
            return bool(self._eval(config.if_))
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

    def __contains__(self, name: str) -> bool:
        """Check if a variable of field exists.

        Args:
            name: name of the field or the variable

        Returns:
            Whether the variable of field exists.
        """
        return name in self._vars or name in self._context

    def __setitem__(self, name: str, value: Any) -> None:  # noqa: ANN401
        """Set a variable or the value of a context object.

        Args:
            name:  The variable or context
            value: The value to assign
        """
        if name in self._context:
            self._context[name].update(value)
        else:
            if not name.isidentifier():
                msg = f"Not a valid variable name: `{name}`"
                raise WorkflowError(msg)
            self._vars[name] = value

    def __getitem__(self, name: str) -> Any:  # noqa: ANN401
        """Get the value of a variable or a context object.

        Args:
            name: the variable name or context object

        Returns:
            The value of the variable or context object.
        """
        if name in self._vars:
            return self._vars[name]
        if name in self._context:
            return self._context[name].value()
        msg = f"Unknown workflow variable or context object: `{name}`"
        raise WorkflowError(msg)


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
    if isinstance(node, ast.Name):
        return True

    return False


class _ReplaceFields(ast.NodeTransformer):
    def __init__(self, workflow: Workflow) -> None:
        self._workflow = workflow
        self.vars: Dict[str, Any] = {}

    def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802
        value = self._workflow[node.id]
        if value is None or isinstance(value, (Number, np.ndarray)):
            self.vars[node.id] = value
            return node
        msg = f"Error in expression: the type of `{node.id}` is not supported"
        raise WorkflowError(msg)
