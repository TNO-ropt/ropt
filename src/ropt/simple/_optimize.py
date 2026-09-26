"""The high-level `optimize` entry point.

One call builds a whole workflow and throws it away again: a context from the
configuration, an evaluator wired to the executor, an optimization step, and the
handlers around it. Nothing survives the call, which is what lets these
functions be called concurrently without any coordination between them.

`optimize_many` is the same thing run several times over, on driver threads,
sharing one executor, whose workers are spread over the runs.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from ropt.components.compute_steps import OptimizationStep
from ropt.components.concurrency import run_concurrent
from ropt.components.event_handlers import ResultsHandler
from ropt.context import EnOptContext

from ._broadcast import (
    broadcast_bundle_sizes,
    broadcast_metadata,
    broadcast_reports,
    broadcast_runs,
)
from ._evaluator import make_evaluator
from ._handlers import attach_handlers
from ._result import OptimizationResult

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any

    from numpy.typing import ArrayLike

    from ropt.components.event_handlers import EventHandler
    from ropt.components.executors import Executor

    from ._function import EvaluationFunction
    from ._report import ReportCallback


def optimize(  # ruff: ignore[too-many-arguments]
    config: dict[str, Any],
    x0: ArrayLike,
    function: EvaluationFunction,
    *,
    executor: Executor | None = None,
    handlers: Sequence[EventHandler] | None = None,
    report: ReportCallback | None = None,
    constraint_tolerance: float = 1e-10,
    bundle_size: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> OptimizationResult:
    """Run a single optimization.

    See [Running Optimizations](../running/running.md) for a walkthrough.

    Without an `executor` the evaluations run in-process, on the calling thread,
    and `bundle_size` does not apply. A closed executor raises a
    [`WorkflowError`][ropt.exceptions.WorkflowError] at the first evaluation. A
    run started from inside an evaluation needs an executor with workers of its
    own: the one it is already running on refuses the work.

    The handlers in `handlers` are called in the order they are listed, and
    the same handler may also be given to other runs, sequential or concurrent,
    to accumulate results across them. A run started from inside a handler must
    not be given that same handler: a nested `optimize` emits on the calling
    thread and raises a [`WorkflowError`][ropt.exceptions.WorkflowError], while
    a nested [`optimize_many`][ropt.simple.optimize_many] emits on its own
    driver threads, which then wait for a lock the calling thread holds.

    Returning `True` from `report` stops the optimization early with
    `USER_ABORT`. Reporting stops there, so results after it in the same batch
    are not passed on.

    Args:
        config:               The optimization configuration.
        x0:                   The initial variable vector.
        function:             The per-realization evaluation function.
        executor:             The executor to evaluate on, or `None`.
        handlers:             Optional handlers, called in the order listed.
        report:               Optional callback invoked per function evaluation.
        constraint_tolerance: The tolerance within which a constraint is satisfied.
        bundle_size:          Evaluations per worker task, `None` for the executor's own.
        metadata:             Optional dictionary attached to every emitted result.

    Returns:
        An [`OptimizationResult`][ropt.simple.OptimizationResult] describing the outcome.
    """
    return _optimize(
        executor,
        config,
        x0,
        function,
        handlers=handlers,
        report=report,
        constraint_tolerance=constraint_tolerance,
        bundle_size=bundle_size,
        metadata=metadata,
    )


def _optimize(  # ruff: ignore[too-many-arguments]
    executor: Executor | None,
    config: dict[str, Any],
    x0: ArrayLike,
    function: EvaluationFunction,
    *,
    handlers: Sequence[EventHandler] | None,
    report: ReportCallback | None,
    constraint_tolerance: float,
    bundle_size: int | None,
    metadata: dict[str, Any] | None = None,
) -> OptimizationResult:
    context = EnOptContext.model_validate(config)
    evaluator = make_evaluator(context, function, executor, bundle_size)
    # This run's own handler, tracking the result the call returns; it is added
    # directly, so it stays out of the handlers the caller manages.
    result_handler = ResultsHandler(constraint_tolerance=constraint_tolerance)
    step = OptimizationStep(evaluator=evaluator)
    step.add_event_handler(result_handler)
    attach_handlers(step, handlers, report)
    exit_code = step.run(
        context=context,
        variables=np.asarray(x0, dtype=np.float64),
        metadata=metadata,
    )
    results = result_handler["results"]
    return OptimizationResult(
        exit_code=exit_code,
        results=None if results is None or results.functions is None else results,
    )


def optimize_many(  # ruff: ignore[too-many-arguments]
    config: dict[str, Any] | Sequence[dict[str, Any]],
    x0: ArrayLike,
    function: EvaluationFunction | Sequence[EvaluationFunction],
    *,
    executor: Executor | None = None,
    handlers: Sequence[EventHandler] | None = None,
    report: ReportCallback | Sequence[ReportCallback] | None = None,
    limit: int | None = None,
    constraint_tolerance: float = 1e-10,
    bundle_size: int | Sequence[int | None] | None = None,
    metadata: dict[str, Any] | Sequence[dict[str, Any]] | None = None,
) -> tuple[OptimizationResult, ...]:
    """Run several optimizations concurrently, sharing one executor.

    Each of `config`, `x0`, and `function` may be a single value (used for
    every run) or a sequence (one per run). Sequences set the number of runs and
    must agree in length; single values are broadcast. A single `x0` is a 1-D
    vector; a per-run sequence of `x0`s is a 2-D matrix with one vector per row.
    A sequence that is empty gives no runs, and returns no results.

    The runs execute concurrently on driver threads and all evaluate on the
    same `executor`, so its workers are shared between them; `limit` bounds how
    many run simultaneously. Without an `executor` the runs still overlap, but
    each evaluation runs in-process on its own driver thread, so `function` is
    then called by several threads at once and must tolerate that. See
    [Parallel Execution and Many Runs](../running/parallel.md#many-optimizations-at-once)
    for a walkthrough, and [Failure in one run](../running/parallel.md#failure-in-one-run)
    for what happens when one raises.

    A handler passed here is fed by every run, since `handle_event` serializes
    its own calls. A handler that combines the events of overlapping runs sees
    them in an order that depends on which run gets there first. A run started
    from inside a handler must not be given that same handler: a nested
    [`optimize`][ropt.simple.optimize] emits on the calling thread and raises a
    [`WorkflowError`][ropt.exceptions.WorkflowError], while a nested
    `optimize_many` emits on its own driver threads, which then wait for a lock
    the calling thread holds. `report=`, being local by nature, is the
    opposite: it is given per run, or broadcast to all of them.

    A closed executor raises a
    [`WorkflowError`][ropt.exceptions.WorkflowError] at the first evaluation. A
    run started from inside an evaluation needs an executor with workers of its
    own: the one it is already running on refuses the work. Returning `True`
    from a `report` callback stops that run early with `USER_ABORT`. `metadata` also reaches
    each run's `function` as `context.metadata`, which makes it a way to tag a
    run, for example with `{"run_id": i}`.

    Args:
        config:               The configuration, or one per run.
        x0:                   The initial variable vector, or one per row.
        function:             The evaluation function, or one per run.
        executor:             The executor every run evaluates on, or `None`.
        handlers:             Optional handlers, fed by every run.
        report:               Optional callback per evaluation, shared or one per run.
        limit:                The maximum number of runs to execute at once.
        constraint_tolerance: The tolerance within which a constraint is satisfied.
        bundle_size:          Evaluations per worker task, shared or one per run.
        metadata:             Optional dictionary attached to every emitted result.

    Returns:
        One [`OptimizationResult`][ropt.simple.OptimizationResult] per run, in order.
    """
    runs = broadcast_runs(config, x0, function)
    reports = broadcast_reports(report, len(runs))
    metadatas = broadcast_metadata(metadata, len(runs))
    bundle_sizes = broadcast_bundle_sizes(bundle_size, len(runs))
    jobs: list[Callable[[], OptimizationResult]] = [
        partial(
            _optimize,
            executor,
            run_config,
            run_x0,
            run_function,
            handlers=handlers,
            report=run_report,
            constraint_tolerance=constraint_tolerance,
            bundle_size=run_bundle_size,
            metadata=run_metadata,
        )
        for (
            (run_config, run_x0, run_function),
            run_report,
            run_metadata,
            run_bundle_size,
        ) in zip(runs, reports, metadatas, bundle_sizes, strict=True)
    ]
    # Dedicated threads, not a shared thread pool: each run blocks its thread
    # while waiting for evaluations that would queue behind it in such a pool.
    return tuple(run_concurrent(jobs, limit))
