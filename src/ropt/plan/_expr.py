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

_SUPPORTED_CONSTANTS: Final = (int, float, bool, str)
_UNARY_OPS: Final = (ast.UAdd, ast.USub, ast.Not)
_BIN_OPS: Final = (ast.Add, ast.Sub, ast.Div, ast.FloorDiv, ast.Mult, ast.Mod, ast.Pow)
_BOOL_OPS: Final = (ast.Or, ast.And)
_CMP_OPS: Final = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
_SUPPORTED_VARIABLE_TYPES: Final = (
    Number,
    str,
    Dict,
    List,
    np.ndarray,
    Results,
    EnOptConfig,
)
_BUILTIN_FUNCTIONS: Final[dict[str, Callable[..., Any]]] = {
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

# A random string used to mark identifiers in expresions:
_PLAN_IDENTIFIER_: Final[str] = "_HHsvLb3BzRHh1JE_"


class ExpressionEvaluator:
    """A class for evaluating mathematical expressions in strings."""

    def __init__(self) -> None:
        """Initialize the expression evaluator.

        Raises:
            ValueError: Raised if any provided function overrides a built-in.
        """
        self._functions = copy.deepcopy(_BUILTIN_FUNCTIONS)

    def add_functions(self, functions: Optional[dict[str, Callable[..., Any]]]) -> None:
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

        2. Strings containing single `$` characters, or enclosed in `$(...)`,
           are evaluated as expressions. Variables and function calls must be
           prefixed by a `$` character. Expressions can access the variables
           passed through the `variables` parameter, a set of built-in
           functions, and any additional functions added using the
           [`add_functions`][ropt.plan.ExpressionEvaluator.add_functions]
           method.

        Tip: Expressions only containing literals
            If an expression has no variables or expressions, you need to enclose
            it in `$(...)` delimiters, like so: `$(1 + 1)`.

        Tip: String Interpolation with `<<...>>`:
            Use the `<<...>>` notation when the overall result should be a
            string. Expressions within `<<...>>` do not require `$(...)` delimiters:
            `"<<1 + 1>>"` will evaluate to `"2"`, but variables and function calls must
            still be prefixed with `$`, for example `"<<$x + 1>>"`.

        Examples of valid expressions, assuming `x == 1` and `y ==2`:

        | Expression    | Result |
        | ------------- | ------ |
        | `$(1 + 1)`    | 2      |
        | `$x`          | 1      |
        | `$x + 1`      | 2      |
        | `$x + $y`     | 3      |
        | `$max(1, $y)` | 2      |

        Info: Supported Features**:
            - Supported Python operators include standard mathematical, boolean,
              and comparison operations.
            - Literal types supported: `int`, `float`, `bool`, and `str`. Nested
              lists (`[...]`) and dictionaries (`{...}`) are supported and can
              reference variables or call functions within them.
            - Allowed types in expressions include numbers, strings, dicts,
              lists, `numpy` arrays, [`Results`][ropt.results.Results], and
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

    def _eval_expr(self, expr: str, variables: Mapping[str, Any]) -> Any:  # noqa: ANN401
        if expr.startswith("$(") and expr.endswith(")"):
            return self._eval_expr(expr[2:-1].strip(), variables)
        expr = re.sub(r"\$(?=[a-zA-Z_][a-zA-Z0-9_]*)", _PLAN_IDENTIFIER_, expr)
        expr = re.sub(r"\${2,}", lambda m: "$" * (len(m.group()) - 1), expr).strip()
        if not expr:
            return ""
        tree = ast.parse(expr, mode="eval")
        if self._is_expression(tree.body):
            transformer = _ExpressionNodeTransformer(variables, self._functions)
            tree = ast.fix_missing_locations(transformer.visit(tree))
            return eval(  # noqa: S307
                compile(tree, "", mode="eval"),
                {"__builtins__": self._functions},
                transformer.values,
            )
        msg = "invalid expression"
        raise SyntaxError(msg)

    def _eval_parts(self, expr: str, variables: Mapping[str, Any]) -> Any:  # noqa: ANN401
        def _substitute(matched: re.Match[str]) -> str:
            parts = matched.string[matched.start() : matched.end()].partition(">>")
            return str(self._eval_expr(parts[0][2:].strip(), variables)) + parts[2]

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

    def _is_expression(  # noqa: C901, PLR0911, PLR0912
        self, node: ast.AST, *, include_ops: bool = True, include_literals: bool = True
    ) -> bool:
        if include_literals:
            if isinstance(node, ast.Constant):
                return node.value is None or type(node.value) in _SUPPORTED_CONSTANTS
            if isinstance(node, ast.List):
                return all(self._is_expression(item) for item in node.elts)
            if isinstance(node, ast.Dict):
                return (
                    all(
                        item is not None and self._is_expression(item)
                        for item in node.keys
                    )
                    and all(self._is_expression(item) for item in node.values)  # noqa: PD011
                )
        if include_ops:
            if isinstance(node, ast.UnaryOp):
                return isinstance(node.op, _UNARY_OPS) and self._is_expression(
                    node.operand
                )
            if isinstance(node, ast.BinOp):
                return (
                    isinstance(node.op, _BIN_OPS)
                    and self._is_expression(node.left)
                    and self._is_expression(node.right)
                )
            if isinstance(node, ast.BoolOp):
                return (
                    isinstance(node.op, _BOOL_OPS)
                    and all(self._is_expression(value) for value in node.values)  # noqa: PD011
                )
            if isinstance(node, ast.Compare):
                return all(isinstance(op, _CMP_OPS) for op in node.ops) and all(
                    self._is_expression(value) for value in node.comparators
                )
        if isinstance(node, ast.Subscript):
            return (
                isinstance(node.value, (ast.Name, ast.Attribute, ast.Subscript))
                and isinstance(node.ctx, ast.Load)
                and self._is_expression(node.slice)
            )
        if isinstance(node, ast.Index):
            return self._is_expression(node.value)  # type: ignore[attr-defined]
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Attribute):
                return self._is_expression(node.value)
            return bool(
                isinstance(node.value, ast.Name) and isinstance(node.ctx, ast.Load)
            )
        if isinstance(node, ast.Call):
            return (
                isinstance(node.func, ast.Name)
                and all(self._is_expression(arg) for arg in node.args)
                and not node.keywords
            )
        return bool(isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load))


class _ExpressionNodeTransformer(ast.NodeTransformer):
    def __init__(
        self,
        variables: Mapping[str, Any],
        functions: dict[str, Callable[..., Any]],
    ) -> None:
        self.values: dict[str, Any] = {}
        self._variables = variables
        self._functions = set(functions.keys())

    def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802
        if node.id.startswith(_PLAN_IDENTIFIER_):
            name = node.id.removeprefix(_PLAN_IDENTIFIER_)
        else:
            msg = f"Invalid expression element: {node.id}."
            if node.id in self._variables or node.id in self._functions:
                msg += " Missing `$`?"
            raise SyntaxError(msg)
        if name in self._variables:
            value = self._variables[name]
            if value is None or isinstance(value, _SUPPORTED_VARIABLE_TYPES):
                self.values[name] = value
                node.id = name
            else:
                msg = f"Error in expression: the type of `{name}` is not supported"
                raise TypeError(msg)
        elif name not in self._functions:
            msg = f"Unknown plan variable or function: `${name}`"
            raise NameError(msg)
        node.id = name
        return self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:  # noqa: N802
        if isinstance(node.value, str):
            node.value = node.value.removeprefix(_PLAN_IDENTIFIER_)
        return node
