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
}


class _ReplaceFields(ast.NodeTransformer):
    def __init__(
        self, variables: Dict[str, Any], functions: Dict[str, Callable[..., Any]]
    ) -> None:
        self.values: Dict[str, Any] = {}
        self._variables = variables
        self._functions = set(functions.keys()) | set(_BUILTIN_FUNCTIONS.keys())

    def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802
        if node.id in self._variables:
            value = self._variables[node.id]
            if value is None or isinstance(value, _SUPPORTED_VARIABLE_TYPES):
                self.values[node.id] = value
                return self.generic_visit(node)
            msg = f"Error in expression: the type of `{node.id}` is not supported"
            raise TypeError(msg)
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:  # noqa: N802
        assert isinstance(node.func, ast.Name)
        if node.func.id not in self._functions:
            msg = f"Unknown plan function: `{node.func.id}`"
            raise AttributeError(msg)
        return self.generic_visit(node)


class ExpressionEvaluator:
    """A class for evaluating mathematical expressions in strings."""

    def __init__(
        self, functions: Optional[Dict[str, Callable[..., Any]]] = None
    ) -> None:
        """Initialize the expression evaluator.

        The `functions` argument allows adding a dictionary of functions that
        can be called within the expression using the `$name()` format. Note
        that these functions cannot override the evaluator's built-in functions.

        Args:
            functions: Optional dictionary of additional functions to add.

        Raises:
            ValueError: Raised if any provided function overrides a built-in.
        """
        self._functions = copy.deepcopy(_BUILTIN_FUNCTIONS)
        if functions is not None:
            for key, value in functions.items():
                if key in _BUILTIN_FUNCTIONS:
                    msg = f"cannot override builtin: `{key}`"
                    raise ValueError(msg)
                self._functions[key] = value

    def eval(
        self,
        expr: str,
        variables: Mapping[str, Any],
    ) -> Any:  # noqa: ANN401
        """Evaluate an expression string, given a dictionary of variable values.

        The `expr` string may take one of the following forms:

        1. Strings enclosed in `${{ ... }}` are evaluated as mathematical
           expressions. Within the expression, variables may be referred to
           using the `$name` format, and functions added to the evaluator may be
           called using the `$function()` format.
        2. Strings enclosed in `$[[ ... ]]` are treated as templates: `$`
           prefixes or `${{ ... }}` expressions within the string are evaluated
           and interpolated as above.
        3. Strings starting with `$` that do not match one of the previous rules
           are evaluated as mathematical expressions. This requires that the
           string starts with a variable or function reference.

        Variables may be referenced in `$name` format. This includes indexing
        using the `[]` operator and attribute access using the `.` operator,
        when appropriate. Multiple `[]` and `.` operators are allowed to an
        arbitrary depth, if supported by the variable. For example, the
        expression `$var['foo'].bar[0]` is valid if `var` contains a dict-like
        value with a `foo` entry that has a `bar` attribute containing a list.

        Info: Supported operators
            A subset of Python operators is supported, including common
            mathematical, boolean, and comparison operators.

        Info: Supported literal types
            Literal types in expressions reflect corresponding Python types.
            Supported literal types include `int`, `float`, `bool`, and `str`.
            Lists (`[ ... ]`) and dictionaries (`{ ... }`) may also be
            constructed and arbitrarily nested, with support for containing
            references to variables and calls to supported functions.

        Info: Supported variable types
            Allowed variable types within expressions include numbers, strings,
            `numpy` arrays, dicts, lists, [`Results`][ropt.results.Results], and
            [`EnOptConfig`][ropt.config.enopt.EnOptConfig] objects.

        Info: Built-in functions
            Functions added at initialization must be prefixed by `$` within
            expressions. However, an exception exists for certain built-in
            functions, where this is optional. For example, the `max` function
            can be called as `$max()` or as `max()` within expressions.

            The following built-in functions are supported: `abs`, `bool`,
            `divmod`, `float`, `int`, `max`, `min`, `pow`, `range`, `round`, and
            `sum`, which map to their corresponding Python built-ins.

        Args:
            expr:      The expression to evaluate.
            variables: A dictionary of variable values.

        Returns:
            The result of the evaluated expression.
        """

        def _substitute(matched: re.Match[str]) -> str:
            return str(
                self.eval(matched.string[matched.start() : matched.end()], variables)
            )

        for variable in variables:
            if variable in self._functions:
                msg = f"conflicting variable/function names: `{variable}`"
                raise ValueError(msg)

        expr = expr.strip()
        if expr.startswith("$$"):
            return expr.replace("$$", "$", 1)
        if expr.startswith("${{") and expr.endswith("}}"):
            return self._eval_expr(expr[3:-2].strip(), variables)
        if expr.startswith("$[[") and expr.endswith("]]"):
            parts = expr[3:-2].split("${{")
            parts[1:] = ["${{" + part for part in parts[1:]]
            return "".join(re.sub(r"\${{(.*)}}", _substitute, part) for part in parts)
        if expr.startswith("$"):
            return self._eval_expr(expr, variables)
        return expr

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
                and all(isinstance(arg, (ast.Name, ast.Constant)) for arg in node.args)
                and not node.keywords
            )
        return bool(isinstance(node, ast.Name))

    def _eval_expr(self, expr: str, variables: Mapping[str, Any]) -> Any:  # noqa: ANN401
        # find all functions and variables:
        found_functions: Dict[str, Callable[..., Any]] = {}
        for word in re.findall(r"(?<=\$)\b(\w+)\b\s*\(", expr):
            if word in self._functions:
                found_functions[word] = self._functions[word]
            else:
                msg = f"Unknown plan function: `{word}`"
                raise AttributeError(msg)
        found_variables: Dict[str, Any] = {}
        for word in re.findall(r"(?<=\$)\b\w+\b", expr):
            if word in variables:
                found_variables[word] = variables[word]
            elif word not in self._functions:
                msg = f"Unknown plan variable: `{word}`"
                raise AttributeError(msg)

        def _substitute(matched: re.Match[str]) -> str:
            value = matched.string[matched.start() : matched.end()]
            if value[1:] in self._functions or value[1:] in variables:
                return value[1:]
            return value

        # Parse the string:
        stripped = re.sub(r"(\$\b\w+\b)", _substitute, expr)
        tree = ast.parse(stripped, mode="eval")
        if self._is_valid(tree.body):
            replacer = _ReplaceFields(found_variables, found_functions)
            tree = ast.fix_missing_locations(replacer.visit(tree))
            return eval(  # noqa: S307
                compile(tree, "", mode="eval"),
                {"__builtins__": self._functions},
                replacer.values,
            )

        msg = "invalid expression"
        raise SyntaxError(msg)
