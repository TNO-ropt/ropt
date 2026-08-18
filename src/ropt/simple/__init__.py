"""The high-level convenience API for running optimizations.

This module builds on the low-level `ropt` primitives. Import its names
directly, for example ``from ropt.simple import optimize, threads``.

Enumerations used in the configuration and results (for example
[`ExitCode`][ropt.enums.ExitCode] and [`VariableType`][ropt.enums.VariableType])
are not re-exported here; import them from [`ropt.enums`][ropt.enums].

An execution block (`threads`, `processes`, `hpc`) and a `handlers` block apply
to the thread that opens them and to the runs this API starts on their behalf.
A run started on a thread you spawn yourself sees neither: it falls back to
evaluating in-process, and its results do not reach the shared handlers. Start
concurrent runs with [`optimize_many`][ropt.simple.optimize_many] instead, which
carries the open blocks to each run.
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

from ._blocks import hpc, processes, threads
from ._evaluate import evaluate, evaluate_many
from ._function import EvaluationFunction
from ._handlers import handlers
from ._offload import can_offload, offload
from ._optimize import optimize, optimize_many
from ._report import ReportCallback
from ._result import EvaluateResult, OptimizeResult

__all__ = [
    "DataFrameHandler",
    "EvaluateResult",
    "EvaluationFunction",
    "EvaluationFunctionContext",
    "EvaluationFunctionResult",
    "EventHandler",
    "HistoryHandler",
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
