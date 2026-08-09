"""The high-level convenience API for running optimizations.

This module builds on the low-level `ropt` primitives. Import its names
directly, for example ``from ropt.simple import optimize, threads``.
"""

from __future__ import annotations

from ropt.components.event_handlers import (
    DataFrameHandler,
    EventHandler,
    HistoryHandler,
    ResultsHandler,
)

from ._evaluate import evaluate, evaluate_many
from ._handlers import handlers
from ._objective import ObjectiveCallback
from ._optimize import optimize, optimize_many
from ._report import ReportCallback
from ._result import EvaluateResult, OptimizeResult
from ._session import hpc, processes, threads

__all__ = [
    "DataFrameHandler",
    "EvaluateResult",
    "EventHandler",
    "HistoryHandler",
    "ObjectiveCallback",
    "OptimizeResult",
    "ReportCallback",
    "ResultsHandler",
    "evaluate",
    "evaluate_many",
    "handlers",
    "hpc",
    "optimize",
    "optimize_many",
    "processes",
    "threads",
]
