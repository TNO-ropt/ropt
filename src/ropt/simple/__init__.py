"""The high-level convenience API for running optimizations.

This module builds on the low-level `ropt` primitives. Import its names
directly, for example ``from ropt.simple import optimize, threads``.

Enumerations used in the configuration and results (for example
[`ExitCode`][ropt.enums.ExitCode] and [`VariableType`][ropt.enums.VariableType])
are not re-exported here; import them from [`ropt.enums`][ropt.enums].
"""

from __future__ import annotations

from ropt.components.evaluators import (
    EvaluationFunctionContext,
    EvaluationFunctionResult,
)
from ropt.components.event_handlers import (
    DataFrameHandler,
    EventHandler,
    HistoryHandler,
    ResultsHandler,
)

from ._evaluate import evaluate, evaluate_many
from ._handlers import handlers
from ._objective import ObjectiveCallback
from ._offload import can_offload, offload
from ._optimize import optimize, optimize_many
from ._report import ReportCallback
from ._result import EvaluateResult, OptimizeResult
from ._session import hpc, processes, threads

__all__ = [
    "DataFrameHandler",
    "EvaluateResult",
    "EvaluationFunctionContext",
    "EvaluationFunctionResult",
    "EventHandler",
    "HistoryHandler",
    "ObjectiveCallback",
    "OptimizeResult",
    "ReportCallback",
    "ResultsHandler",
    "can_offload",
    "evaluate",
    "evaluate_many",
    "handlers",
    "hpc",
    "offload",
    "optimize",
    "optimize_many",
    "processes",
    "threads",
]
