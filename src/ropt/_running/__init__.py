"""Implementation of the optimization runs exposed by the `ropt` package.

The names defined here are re-exported by [`ropt`][ropt] and imported from
there.
"""

from __future__ import annotations

from ._function import EvaluationFunction
from ._functions import evaluate, evaluate_batch, optimize, run
from ._pool import WorkerPool
from ._result import EvaluationResult, OptimizationResult
from ._result_callback import ResultCallback
from ._session import Session, session

__all__ = [
    "EvaluationFunction",
    "EvaluationResult",
    "OptimizationResult",
    "ResultCallback",
    "Session",
    "WorkerPool",
    "evaluate",
    "evaluate_batch",
    "optimize",
    "run",
    "session",
]
