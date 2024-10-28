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
    def __init__(
        self, functions: Optional[Dict[str, Callable[..., Any]]] = None
    ) -> None:
        self._functions = copy.deepcopy(_BUILTIN_FUNCTIONS)
        if functions is not None:
            for key, value in functions.items():
                if key in _BUILTIN_FUNCTIONS:
                    msg = f"cannot override builtin: `{key}`"
                    raise ValueError(msg)
                self._functions[key] = value

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

    def eval(
        self,
        value: str,
        variables: Mapping[str, Any],
    ) -> Any:  # noqa: ANN401
        def _substitute(matched: re.Match[str]) -> str:
            value = matched.string[matched.start() : matched.end()]
            return "$" if value == "$$" else str(self.eval(value, variables))

        for variable in variables:
            if variable in self._functions:
                msg = f"conflicting variable/function names: `{variable}`"
                raise ValueError(msg)

        value = value.strip()
        if value.startswith("$$"):
            return value.replace("$$", "$", 1)
        if value.startswith("${{") and value.endswith("}}"):
            return self._eval_expr(value[3:-2].strip(), variables)
        if value.startswith("$[[") and value.endswith("]]"):
            parts = value[3:-2].split("${{")
            parts[1:] = ["${{" + part for part in parts[1:]]
            return "".join(
                re.sub(r"\${{(.*)}}|\$\$|\$([^\W0-9][\w\.]*)", _substitute, part)
                for part in parts
            )
        if value.startswith("$"):
            return self._eval_expr(value, variables)
        return value
