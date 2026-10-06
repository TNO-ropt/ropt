"""The main `ropt` module, a library for ensemble based optimization.

Import the names used to run an optimization directly from this package, for
example `from ropt import optimize, session`. See
[Running Optimizations](../running/running.md) for a walkthrough.

A name is re-exported here when a program written against `ropt` has to spell it
out: the functions it calls, the classes it instantiates or subclasses, and the
types it puts in its own signatures. A name that is only reached through a value
a run returns keeps its own module. The fields of a
[`FunctionResults`][ropt.results.FunctionResults] are in
[`ropt.results`][ropt.results], the remaining enumerations in
[`ropt.enums`][ropt.enums], and the classes describing the configuration
dictionary in [`ropt.config`][ropt.config].

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
from ropt.enums import ExitCode
from ropt.results import (
    FunctionResults,
    GradientResults,
    Results,
    results_to_pandas,
    results_to_polars,
)
from ropt.run import (
    EvaluationFunction,
    EvaluationResult,
    OptimizationResult,
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
    "ExitCode",
    "FunctionResults",
    "GradientResults",
    "HistoryHandler",
    "OptimizationResult",
    "Results",
    "ResultsHandler",
    "Session",
    "WorkerPool",
    "evaluate",
    "evaluate_batch",
    "optimize",
    "optimize_many",
    "results_to_pandas",
    "results_to_polars",
    "session",
]
