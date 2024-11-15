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
            if isinstance(value, np.ndarray):
                value = value.tolist()
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

        The `expr` string is evaluated according to the following rules:

        1. Strings containing substrings delimited by `<<...>>` are treated as
           templates. The content of each `<<...>>` instance is evaluated as an
           expression, and the result replaces the `<<...>>` substring.

        2. Strings containing a single `$` character, or enclosed in `<<...>>`,
           are evaluated as expressions. Then, singular `$` characters are
           removed and sequences of two or more `$` characters are shortened by
           one. The remaining string is then evaluated as a Python expression
           with limited functionality. These expressions can access the
           variables passed through the `variables` parameter, a set of built-in
           functions, and any additional functions added using the
           [`add_functions`][ropt.plan.ExpressionEvaluator.add_functions]
           method.

        Tip: String Interpolation with `<<...>>`:
            Use the `<<...>>` notation when the overall result should be a
            string. Expressions within `<<...>>` do not require `$` signs,
            though it may be clearer to prefix variables with `$`, for example:
            `<<$x + 1>>`

        Recommended Practices:
            - Prefix variables with `$` for clarity.
            - Optionally prefix function calls with `$`, in particular if the function
              is not a Python built-in function.
            - Enclose expressions without variables or functions in `$(...)`.

        Examples of valid expressions:
        ```python
        $(1 + 1)
        $x
        $x + 1
        $x + $y
        1 + $x
        $max(1, $x)
        ```

        Omitting some `$` signs is valid, and may enhance clarity:
        ```python
        $max(1, x)
        max(1, $x)
        ```

        Less recommended, but still valid:
        ```python
        $1 + 1
        $(x)
        $1 + $x
        $1 + x
        $(x + 1)
        $($x + 1)
        $x + y
        ```

        Info: Supported Features**:
            - Supported Python operators include standard mathematical, boolean,
              and comparison operations.
            - Literal types supported: `int`, `float`, `bool`, and `str`. Nested
              lists (`[...]`) and dictionaries (`{...}`) are supported and can
              reference variables or call functions within them.
            - Allowed types in expressions include numbers, strings, dicts,
              lists, [`Results`][ropt.results.Results], and
              [`EnOptConfig`][ropt.config.enopt.EnOptConfig] objects.
            - Built-in functions include: `abs`, `bool`, `divmod`, `float`,
              `int`, `max`, `min`, `pow`, `range`, `round`, `sum`, `str`.

            Plan variables may contain objects of any type. However, to be used
            in an expression, they should contain a value of a supported type or
            be convertible to a supported type before evaluation. In particular,
            `numpy` arrays are converted to lists when encountered in an expression.

        Args:
            expr:      The expression to evaluate, provided as a string.
            variables: A dictionary containing the variable values.

        Returns:
            The result of evaluating the expression.
        """
        for variable in variables:
            if variable in self._functions:
                msg = f"conflicting variable/function names: `{variable}`"
                raise ValueError(msg)

        expr = expr.strip()
        if "<<" in expr:
            return self._eval_parts(expr, variables)
        if bool(re.search(r"(?<!\$)\$(?!\$)", expr)):
            return self._eval_expr(expr, variables)
        return re.sub(r"\${2,}", lambda m: "$" * (len(m.group()) - 1), expr)

    def _eval_parts(self, expr: str, variables: Mapping[str, Any]) -> Any:  # noqa: ANN401
        def _substitute(matched: re.Match[str]) -> str:
            parts = matched.string[matched.start() : matched.end()].partition(">>")
            return str(self._eval_expr(parts[0][2:], variables)) + parts[2]

        parts = []
        prefix = ""
        for part in re.split("(<{2,})", expr):
            if part.endswith("<<"):
                parts.append(part[:-2])
                prefix = "<<"
            else:
                parts.append(prefix + part)
                prefix = ""
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
        stripped = re.sub(r"(?<!\$)\$(?!\$)", "", expr)
        stripped = re.sub(
            r"\${2,}", lambda m: "$" * (len(m.group()) - 1), stripped
        ).strip()
        if not stripped:
            return ""
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
