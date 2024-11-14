"""This module defines the optimization plan object."""

from __future__ import annotations

import ast
import copy
import re
from numbers import Number
from typing import Any, Callable, Dict, Final, List, Mapping, Optional

import numpy as np

from ropt.config.enopt import EnOptConfig
from ropt.results import Results

_VALID_TYPES: Final = (int, float, bool, str)
_UNARY_OPS: Final = (ast.UAdd, ast.USub, ast.Not)
_BIN_OPS: Final = (ast.Add, ast.Sub, ast.Div, ast.FloorDiv, ast.Mult, ast.Mod, ast.Pow)
_BOOL_OPS: Final = (ast.Or, ast.And)
_CMP_OPS: Final = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)

_SUPPORTED_VARIABLE_TYPES: Final = (
    Number,
    str,
    np.ndarray,
    Dict,
    List,
    Results,
    EnOptConfig,
)
_BUILTIN_FUNCTIONS: Final[Dict[str, Callable[..., Any]]] = {
    "abs": abs,
    "bool": bool,
    "divmod": divmod,
    "float": float,
    "int": int,
    "max": max,
    "min": min,
    "pow": pow,
    "range": range,
    "round": round,
    "sum": sum,
    "str": str,
}


class _ReplaceFields(ast.NodeTransformer):
    def __init__(
        self, variables: Mapping[str, Any], functions: Dict[str, Callable[..., Any]]
    ) -> None:
        self.values: Dict[str, Any] = {}
        self._variables = variables
        self._functions = set(functions.keys())

    def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802
        if node.id in self._variables:
            value = self._variables[node.id]
            if value is None or isinstance(value, _SUPPORTED_VARIABLE_TYPES):
                self.values[node.id] = value
                return self.generic_visit(node)
            msg = f"Error in expression: the type of `{node.id}` is not supported"
            raise TypeError(msg)
        if node.id not in self._functions:
            msg = f"Unknown plan variable: `{node.id}`"
            raise AttributeError(msg)
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:  # noqa: N802
        assert isinstance(node.func, ast.Name)
        if node.func.id not in self._functions:
            msg = f"Unknown plan function: `{node.func.id}`"
            raise AttributeError(msg)
        return self.generic_visit(node)


class ExpressionEvaluator:
    """A class for evaluating mathematical expressions in strings."""

    def __init__(self) -> None:
        """Initialize the expression evaluator.

        Raises:
            ValueError: Raised if any provided function overrides a built-in.
        """
        self._functions = copy.deepcopy(_BUILTIN_FUNCTIONS)
        self._functions["_eval"] = self._eval_function

    def add_functions(self, functions: Optional[Dict[str, Callable[..., Any]]]) -> None:
        """Add functions to the evaluator.

        Args:
            functions: A dictionary of functions to add.

        Raises:
            ValueError: If a function already exists.
        """
        if functions is not None:
            for key in functions:
                if key in self._functions:
                    msg = f"cannot override existing function: `{key}`"
                    raise ValueError(msg)
            for key, value in functions.items():
                self._functions[key] = value

    def eval(
        self,
        expr: str,
        variables: Mapping[str, Any],
    ) -> Any:  # noqa: ANN401
        """Evaluate an expression string with provided variable values.

        The `expr` string may take one of the following forms:

        1. Strings enclosed in `$(...)` are evaluated as expressions, that may
           contain variables references and nested calls to functions. Functions
           must be built-in to the expression evaluator, or have been added to
           the evaluator object when it was constructed.

        2. Strings of the form `$name(...)` are evaluated as function calls. The
           contents between the `()` brackets may be an expression that is
           evaluated before passing the result to the function. An expression
           of the form `$name(...)` is equivalent to an expression evaluation of
           the form `$(name(...))`.

        3. Strings starting with `$name`, where name is not a function, i.e. not
           of the form `$name(...)`, are evaluated as variable references. This
           syntax supports nested indexing via `[]` and attribute access with
           `.`. For example, `$var['foo'].bar[0]` is valid if `var` contains a
           compatible structure like a dictionary. Indices within `[]` may
           themselves be expressions that refer to other variables and
           functions. An expression of the form `$name...` is equivalent to an
           expression evaluation of the form `$(name...), where `...` can be any
           combination of indexing and attribute access.

        4. Strings in @(...) are treated as templates. Within these, substrings
           delimited by `<<...>>` are evaluated as if enclosed in `$(...)` and
           the result substituted for the `<<...>>` substring.

        Info: Supported features
            - A subset of Python operators is supported, covering standard
              mathematical, boolean, and comparison operations.

            - Literals include `int`, `float`, `bool`, and `str`. Nested lists
              (`[...]`) and dictionaries (`{...}`) are supported and can reference
              variables or call functions within their content.

            - Allowed types within expressions include numbers, strings, `numpy`
              arrays, dicts, lists, [`Results`][ropt.results.Results], and
              [`EnOptConfig`][ropt.config.enopt.EnOptConfig] objects.

            - Supported built-ins include: `abs`, `bool`, `divmod`, `float`,
              `int`, `max`, `min`, `pow`, `range`, `round`, `sum`, `str`.

        Note: Implicit use of `$(...)` and `@(...)`
            Where appropriate, the `$(...)` and `@(...)` delimiters might be
            implicit in some cases.

            For instance, plan steps support an `if` attribute in their
            configuration allowing for conditional execution. This can be
            specified using the `$(...)` syntax, but this is optional. For
            example, instead of `"if": "${x > 0}"`, `"if": "x > 0"` may be used.
            (See [`PlanStepConfig`][ropt.config.plan.PlanStepConfig]).

            Similarly, the [`print`][ropt.plugins.plan._print.DefaultPrintStep]
            step is an example where the `@(...)` delimiters are optional. This
            step prints a message that will be evaluated, thereby substituting
            any occurrences of `<<...>>`. However, the use of surrounding
            `@(...)` delimiters in the message attribute of the step
            configuration is optional, and they will be implicitly added if not
            present.

        Args:
            expr:      The expression to evaluate as a string.
            variables: A dictionary of variable values.

        Returns:
            The evaluated result.
        """
        for variable in variables:
            if variable in self._functions:
                msg = f"conflicting variable/function names: `{variable}`"
                raise ValueError(msg)

        expr = expr.strip()
        if expr.startswith("$$"):
            return expr.replace("$$", "$", 1)
        if expr.startswith("@(") and expr.endswith(")"):
            return self._eval_parts(expr[2:-1], variables)
        if expr.startswith("$("):
            return self._eval_expr("_eval" + expr[1:], variables)
        if expr.startswith("$"):
            self._check_variable_or_function(expr[1:])
            return self._eval_expr(expr[1:], variables)
        return expr

    def _eval_parts(self, expr: str, variables: Mapping[str, Any]) -> Any:  # noqa: ANN401
        def _substitute(matched: re.Match[str]) -> str:
            parts = matched.string[matched.start() : matched.end()].partition(">>")
            return str(self._eval_expr(parts[0][2:], variables)) + parts[2]

        parts = expr.split("<<")
        for part in parts[1:]:
            if part.count(">>") != 1:
                msg = "Missing, or too many, `>>` in @() expression"
                raise SyntaxError(msg)
        parts[1:] = ["<<" + part for part in parts[1:]]
        return "".join(re.sub(r"<<(.*)>>", _substitute, part) for part in parts)

    def _eval_function(self, value: Any) -> Any:  # noqa: ANN401
        return value

    def _is_valid(self, node: ast.AST) -> bool:  # noqa: C901, PLR0911
        if isinstance(node, ast.Constant):
            return node.value is None or type(node.value) in _VALID_TYPES
        if isinstance(node, ast.UnaryOp):
            return isinstance(node.op, _UNARY_OPS) and self._is_valid(node.operand)
        if isinstance(node, ast.BinOp):
            return (
                isinstance(node.op, _BIN_OPS)
                and self._is_valid(node.left)
                and self._is_valid(node.right)
            )
        if isinstance(node, ast.BoolOp):
            return (
                isinstance(node.op, _BOOL_OPS)
                and all(self._is_valid(value) for value in node.values)  # noqa: PD011
            )
        if isinstance(node, ast.Compare):
            return all(isinstance(op, _CMP_OPS) for op in node.ops) and all(
                self._is_valid(value) for value in node.comparators
            )
        if isinstance(node, ast.List):
            return all(self._is_valid(item) for item in node.elts)
        if isinstance(node, ast.Dict):
            return (
                all(item is None or self._is_valid(item) for item in node.keys)
                and all(self._is_valid(item) for item in node.values)  # noqa: PD011
            )
        if isinstance(node, ast.Subscript):
            return self._is_valid(node.slice)
        if isinstance(node, ast.Index):
            return self._is_valid(node.value)  # type: ignore[attr-defined]
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Attribute):
                return self._is_valid(node.value)
            return bool(isinstance(node.value, ast.Name))
        if isinstance(node, ast.Call):
            return (
                isinstance(node.func, ast.Name)
                and all(self._is_valid(arg) for arg in node.args)
                and not node.keywords
            )
        return bool(isinstance(node, ast.Name))

    def _eval_expr(self, expr: str, variables: Mapping[str, Any]) -> Any:  # noqa: ANN401
        stripped = expr.strip()
        tree = ast.parse(stripped, mode="eval")
        if self._is_valid(tree.body):
            replacer = _ReplaceFields(variables, self._functions)
            tree = ast.fix_missing_locations(replacer.visit(tree))
            return eval(  # noqa: S307
                compile(tree, "", mode="eval"),
                {"__builtins__": self._functions},
                replacer.values,
            )

        msg = "invalid expression"
        raise SyntaxError(msg)

    def _is_valid_variable_or_function(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Subscript):
            return self._is_valid(node.slice)
        if isinstance(node, ast.Index):
            return self._is_valid(node.value)  # type: ignore[attr-defined]
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Attribute):
                return self._is_valid(node.value)
            return bool(isinstance(node.value, ast.Name))
        if isinstance(node, ast.Call):
            return (
                isinstance(node.func, ast.Name)
                and all(self._is_valid(arg) for arg in node.args)
                and not node.keywords
            )
        return bool(isinstance(node, ast.Name))

    def _check_variable_or_function(self, expr: str) -> Any:  # noqa: ANN401
        stripped = expr.strip()
        tree = ast.parse(stripped, mode="eval")
        if not self._is_valid_variable_or_function(tree.body):
            msg = "invalid variable or function"
            raise SyntaxError(msg)
