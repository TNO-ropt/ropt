"""The high-level convenience API for running optimizations.

This module builds on the low-level `ropt` primitives. Import its names
directly, for example `from ropt.simple import optimize, ProcessExecutor`. See
[Running Optimizations](../running/running.md) for a walkthrough.

Enumerations used in the configuration and results (for example
[`ExitCode`][ropt.enums.ExitCode] and [`VariableType`][ropt.enums.VariableType])
are not re-exported here; import them from [`ropt.enums`][ropt.enums].

Nothing about a run depends on where it is called from. Where its evaluations
happen is decided by the executor it is given with `executor=`, and which
handlers see its results by the `handlers=` it is given. A run given no executor
evaluates in-process. This holds wherever the run is started from, including a
thread you spawn yourself.
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
from ropt.components.executors import (
    Executor,
    HPCExecutor,
    LocalJobExecutor,
    ProcessExecutor,
    ThreadExecutor,
)

from ._evaluate import evaluate, evaluate_many
from ._function import EvaluationFunction
from ._offload import offload
from ._optimize import optimize, optimize_many
from ._report import ReportCallback
from ._result import OptimizationResult

__all__ = [
    "DataFrameHandler",
    "EvaluationFunction",
    "EvaluationFunctionContext",
    "EvaluationFunctionResult",
    "EventHandler",
    "Executor",
    "HPCExecutor",
    "HistoryHandler",
    "LocalJobExecutor",
    "OptimizationResult",
    "ProcessExecutor",
    "ReportCallback",
    "ResultsHandler",
    "ThreadExecutor",
    "evaluate",
    "evaluate_many",
    "offload",
    "optimize",
    "optimize_many",
]
