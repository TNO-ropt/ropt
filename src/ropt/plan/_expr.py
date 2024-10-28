"""This module defines the optimization plan object."""

from __future__ import annotations

import ast
import re
from numbers import Number
from typing import (
    Any,
    Dict,
    Final,
    List,
    Mapping,
)

import numpy as np

from ropt.config.enopt import EnOptConfig
from ropt.results import Results

_VALID_TYPES: Final = (int, float, bool, str)
_UNARY_OPS: Final = (ast.UAdd, ast.USub, ast.Not)
_BIN_OPS: Final = (ast.Add, ast.Sub, ast.Div, ast.FloorDiv, ast.Mult, ast.Mod, ast.Pow)
_BOOL_OPS: Final = (ast.Or, ast.And)
_CMP_OPS: Final = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)

_SUPPORTED_VARIABLE_TYPES = (Number, str, np.ndarray, Dict, List, Results, EnOptConfig)


class _ReplaceFields(ast.NodeTransformer):
    def __init__(self, variables: Dict[str, Any]) -> None:
        self.values: Dict[str, Any] = {}
        self._variables = variables

    def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802
        if node.id in self._variables:
            value = self._variables[node.id]
            if value is None or isinstance(value, _SUPPORTED_VARIABLE_TYPES):
                self.values[node.id] = value
                return self.generic_visit(node)
            msg = f"Error in expression: the type of `{node.id}` is not supported"
            raise TypeError(msg)
        return self.generic_visit(node)


class ExpressionEvaluator:
    def __init__(self, variables: Mapping[str, Any]) -> None:
        self._variables = variables

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
        return bool(isinstance(node, ast.Name))

    def _eval_expr(self, expr: str) -> Any:  # noqa: ANN401
        # find all variables:
        found: Dict[str, Any] = {}
        for word in re.findall(r"(?<=\$)\b\w+\b", expr):
            if word not in self._variables:
                msg = f"Unknown plan variable: `{word}`"
                raise AttributeError(msg)
            found[word] = self._variables[word]

        # Parse the string:
        stripped = re.sub(r"\$(\$*)", "\\1", expr)
        tree = ast.parse(stripped, mode="eval")
        if self._is_valid(tree.body):
            replacer = _ReplaceFields(found)
            tree = ast.fix_missing_locations(replacer.visit(tree))
            return eval(  # noqa: S307
                compile(tree, "", mode="eval"), {"__builtins__": {}}, replacer.values
            )

        msg = "invalid expression"
        raise SyntaxError(msg)

    def _substitute(self, matched: re.Match[str]) -> str:
        value = matched.string[matched.start() : matched.end()]
        return "$" if value == "$$" else str(self.eval(value))

    def eval(self, value: str) -> Any:  # noqa: ANN401
        value = value.strip()
        if value.startswith("$$"):
            return value.replace("$$", "$", 1)
        if value.startswith("{{") and value.endswith("}}"):
            return self._eval_expr(value[2:-2].strip())
        if value.startswith("[[") and value.endswith("]]"):
            parts = value[2:-2].split("{{")
            parts[1:] = ["{{" + part for part in parts[1:]]
            return "".join(
                re.sub(r"\{{(.*)}}|\$\$|\$([^\W0-9][\w\.]*)", self._substitute, part)
                for part in parts
            )
        if value.startswith("$"):
            return self._eval_expr(value)
        return value
