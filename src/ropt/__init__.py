"""The main `ropt` module, a library for ensemble based optimization.

Import the names used to run an optimization directly from this package, for
example `from ropt import optimize, session`. See
[Running Optimizations](../running/running.md) for a walkthrough.

A name is re-exported here when a program written against `ropt` has to spell it
out: the functions it calls, the classes it instantiates or subclasses, and the
types it puts in its own signatures. A name that is only reached through a value
an optimization or evaluation returns keeps its own module. The fields of a
[`FunctionResults`][ropt.results.FunctionResults] are in
[`ropt.results`][ropt.results], the remaining enumerations in
[`ropt.enums`][ropt.enums], and the classes describing the configuration
dictionary in [`ropt.config`][ropt.config].

Exceptions follow the same rule. Every error that can reach a caller derives
from [`RoptError`][ropt.exceptions.RoptError] and is re-exported here; the
control-flow signals that `ropt` raises and catches itself stay in
[`ropt.exceptions`][ropt.exceptions].

Nothing about an optimization or evaluation depends on where it is called from.
The functions [`optimize`][ropt.optimize], [`evaluate`][ropt.evaluate],
[`evaluate_batch`][ropt.evaluate_batch] and [`run`][ropt.run] take an optional
`pool` argument that says where the work happens: without one it runs
in-process on the calling thread, with a [`WorkerPool`][ropt.WorkerPool] it runs
on that pool's workers and belongs to the pool's session. This holds wherever
the call is made from, including a thread you spawn yourself.

Pools come from a [`session`][ropt.session], which releases them when it closes.
"""
# ruff: file-ignore[non-empty-init-module]

from __future__ import annotations

import logging

from ropt.components.concurrency import aborting, run_program
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
from ropt.exceptions import (
    AbortedError,
    ExecutionError,
    RoptError,
    UnsupportedError,
    WorkflowError,
)
from ropt.results import (
    FunctionResults,
    GradientResults,
    Results,
    results_to_pandas,
    results_to_polars,
)
from ropt.running import (
    EvaluationFunction,
    EvaluationResult,
    OptimizationResult,
    ReportCallback,
    Session,
    WorkerPool,
    evaluate,
    evaluate_batch,
    optimize,
    run,
    session,
)

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "AbortedError",
    "DataFrameHandler",
    "EvaluationFunction",
    "EvaluationFunctionContext",
    "EvaluationFunctionResult",
    "EvaluationResult",
    "EventHandler",
    "ExecutionError",
    "ExitCode",
    "FunctionResults",
    "GradientResults",
    "HistoryHandler",
    "OptimizationResult",
    "ReportCallback",
    "Results",
    "ResultsHandler",
    "RoptError",
    "Session",
    "UnsupportedError",
    "WorkerPool",
    "WorkflowError",
    "aborting",
    "evaluate",
    "evaluate_batch",
    "optimize",
    "results_to_pandas",
    "results_to_polars",
    "run",
    "run_program",
    "session",
]
