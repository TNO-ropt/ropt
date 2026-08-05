"""The high-level convenience API for running optimizations.

This subpackage builds on top of the [`ropt.workflow`][ropt.workflow]
primitives. Its public names are lazily re-exported at the top level
(`ropt.optimize`, ...) for convenience.
"""

from __future__ import annotations

from ._evaluate import evaluate, evaluate_many
from ._objective import ObjectiveCallback
from ._optimize import optimize, optimize_many
from ._result import EvaluateResult, OptimizeResult
from ._session import processes, threads

__all__ = [
    "EvaluateResult",
    "ObjectiveCallback",
    "OptimizeResult",
    "evaluate",
    "evaluate_many",
    "optimize",
    "optimize_many",
    "processes",
    "threads",
]
