"""The main `ropt` module, a library for ensemble based optimization.

Import the names used to run an optimization directly from this package, for
example `from ropt import optimize, session`. See
[Running Optimizations](../running/running.md) for a walkthrough.

Enumerations used in the configuration and results (for example
[`ExitCode`][ropt.enums.ExitCode] and
[`VariableType`][ropt.enums.VariableType]) are not re-exported here; import them
from [`ropt.enums`][ropt.enums].

Nothing about a run depends on where it is called from. What it is started on
says where its evaluations happen and which session it belongs to: a module
function runs in-process and belongs to nothing, a [`Session`][ropt.Session]
method runs in-process on that session, and a [`WorkerPool`][ropt.WorkerPool]
method runs on that pool's workers. This holds wherever the run is started from,
including a thread you spawn yourself.

Pools come from a [`session`][ropt.session], which releases them when it closes.
"""
# ruff: file-ignore[non-empty-init-module]

from __future__ import annotations

import logging

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
from ropt.run import (
    EvaluationFunction,
    EvaluationResult,
    OptimizationResult,
    ReportCallback,
    Session,
    WorkerPool,
    evaluate,
    evaluate_batch,
    optimize,
    optimize_many,
    session,
)

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "DataFrameHandler",
    "EvaluationFunction",
    "EvaluationFunctionContext",
    "EvaluationFunctionResult",
    "EvaluationResult",
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
