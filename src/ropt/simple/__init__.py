"""The high-level convenience API for running optimizations.

This module builds on the low-level `ropt` primitives. Import its names
directly, for example `from ropt.simple import optimize, session`. See
[Running Optimizations](../running/running.md) for a walkthrough.

Enumerations used in the configuration and results (for example
[`ExitCode`][ropt.enums.ExitCode] and [`VariableType`][ropt.enums.VariableType])
are not re-exported here; import them from [`ropt.enums`][ropt.enums].

Nothing about a run depends on where it is called from. What it is started on
says where its evaluations happen and which session it belongs to: a module
function runs in-process and belongs to nothing, a
[`Session`][ropt.simple.Session] method runs in-process on that session, and a
[`WorkerPool`][ropt.simple.WorkerPool] method runs on that pool's workers. This
holds wherever the run is started from, including a thread you spawn yourself.

Pools come from a [`session`][ropt.simple.session], which releases them when it
closes.
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

from ._aborted import ABORTED, Aborted
from ._function import EvaluationFunction
from ._functions import evaluate, evaluate_batch, optimize, optimize_many
from ._pool import WorkerPool
from ._report import ReportCallback
from ._result import OptimizationResult
from ._session import Session, session

__all__ = [
    "ABORTED",
    "Aborted",
    "DataFrameHandler",
    "EvaluationFunction",
    "EvaluationFunctionContext",
    "EvaluationFunctionResult",
    "EventHandler",
    "HistoryHandler",
    "OptimizationResult",
    "ReportCallback",
    "ResultsHandler",
    "Session",
    "WorkerPool",
    "evaluate",
    "evaluate_batch",
    "optimize",
    "optimize_many",
    "session",
]
