"""The high-level `optimize` entry point.

One call builds a whole workflow and throws it away again: a context from the
configuration, an evaluator wired to the pool, an optimization step, and the
handlers around it. Nothing survives the call, which is what lets these
functions be called concurrently without any coordination between them.

`optimize_many` is the same thing run several times over, on driver threads,
sharing one pool. Sharing the pool is what makes its runs cooperate: they draw
their batch IDs from one counter and send their evaluations to the same workers.
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
from ._guards import check_pool
from ._handlers import attach_handlers
from ._pool import serial_pool
from ._result import OptimizationResult

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any

    from numpy.typing import ArrayLike

    from ropt.components.event_handlers import EventHandler

    from ._function import EvaluationFunction
    from ._pool import WorkerPool
    from ._report import ReportCallback


def optimize(  # ruff: ignore[too-many-arguments]
    config: dict[str, Any],
    x0: ArrayLike,
    function: EvaluationFunction,
    *,
    pool: WorkerPool | None = None,
    handlers: Sequence[EventHandler] | None = None,
    report: ReportCallback | None = None,
    constraint_tolerance: float = 1e-10,
    bundle_size: int = 1,
    metadata: dict[str, Any] | None = None,
) -> OptimizationResult:
    """Run a single optimization.

    See [Running Optimizations](../running/running.md) for a walkthrough.

    A pool or group that is closed — because it was closed directly, or
    because its session ended — is refused here with a
    [`WorkflowError`][ropt.exceptions.WorkflowError], as is one carried into
    a worker process, where it cannot work at all.

    Without a `pool` the evaluations run in-process, on the calling thread. A
    run started from inside an evaluation needs a pool with workers of its own:
    the pool it is already running on refuses the work. A
    [`serial_pool`][ropt.simple.serial_pool] evaluates inline and has no workers
    to occupy, so it can be reused.

    The handlers in `handlers` are called in the order they are listed, and
    the same handler may also be given to other runs, sequential or concurrent,
    to accumulate results across them.

    Returning `True` from `report` stops the optimization early with
    `USER_ABORT`. Reporting stops there, so results after it in the same batch
    are not passed on.

    Args:
        config:               The optimization configuration.
        x0:                   The initial variable vector.
        function:             The per-realization evaluation function.
        pool:                 The pool to evaluate on, from a session factory.
        handlers:             Optional handlers, called in the order listed.
        report:               Optional callback invoked per function evaluation.
        constraint_tolerance: The tolerance within which a constraint is satisfied.
        bundle_size:          Evaluations per worker task, `0` for a whole batch.
        metadata:             Optional dictionary attached to every emitted result.

    Returns:
        An [`OptimizationResult`][ropt.simple.OptimizationResult] describing the outcome.
    """
    check_pool(pool)
    return _optimize(
        pool if pool is not None else serial_pool(),
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
    pool: WorkerPool,
    config: dict[str, Any],
    x0: ArrayLike,
    function: EvaluationFunction,
    *,
    handlers: Sequence[EventHandler] | None,
    report: ReportCallback | None,
    constraint_tolerance: float,
    bundle_size: int,
    metadata: dict[str, Any] | None = None,
) -> OptimizationResult:
    context = EnOptContext.model_validate(config)
    evaluator = make_evaluator(context, function, pool, bundle_size)
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
    pool: WorkerPool | None = None,
    handlers: Sequence[EventHandler] | None = None,
    report: ReportCallback | Sequence[ReportCallback] | None = None,
    limit: int | None = None,
    constraint_tolerance: float = 1e-10,
    bundle_size: int | Sequence[int] = 1,
    metadata: dict[str, Any] | Sequence[dict[str, Any]] | None = None,
) -> tuple[OptimizationResult, ...]:
    """Run several optimizations concurrently, sharing one pool.

    Each of `config`, `x0`, and `function` may be a single value (used for
    every run) or a sequence (one per run). Sequences set the number of runs and
    must agree in length; single values are broadcast. A single `x0` is a 1-D
    vector; a per-run sequence of `x0`s is a 2-D matrix with one vector per row.
    A sequence that is empty gives no runs, and returns no results.

    The runs execute concurrently on driver threads and all evaluate on the
    same `pool`, so its workers are shared between them; `limit` bounds how
    many run simultaneously. Without a `pool` the runs still overlap, but each
    evaluation runs in-process on its own driver thread, so `function` is then
    called by several threads at once and must tolerate that. See
    [Parallel Execution and Many Runs](../running/parallel.md#many-optimizations-at-once)
    for a walkthrough, and [Failure in one run](../running/parallel.md#failure-in-one-run)
    for what happens when one raises.

    A handler passed here is fed by every run, since `handle_event` serializes
    its own calls. A handler that combines the events of overlapping runs sees
    them in an order that depends on which run gets there first. `report=`,
    being local by nature, is the opposite: it is given per run, or broadcast
    to all of them.

    A pool that is closed — because it was closed directly, or because its
    session ended — is refused here with a
    [`WorkflowError`][ropt.exceptions.WorkflowError], as is one carried into
    a worker process, where it cannot work at all.

    A run started from inside an evaluation needs a pool with workers of its
    own: the pool it is already running on refuses the work. A
    [`serial_pool`][ropt.simple.serial_pool] evaluates inline and has no workers
    to occupy, so it can be reused. Returning `True` from a `report`
    callback stops that run early with `USER_ABORT`. `metadata` also reaches
    each run's `function` as `context.metadata`, which makes it a way to tag a
    run, for example with `{"run_id": i}`.

    Args:
        config:               The configuration, or one per run.
        x0:                   The initial variable vector, or one per row.
        function:             The evaluation function, or one per run.
        pool:                 The pool every run evaluates on.
        handlers:             Optional handlers, fed by every run.
        report:               Optional callback per evaluation, shared or one per run.
        limit:                The maximum number of runs to execute at once.
        constraint_tolerance: The tolerance within which a constraint is satisfied.
        bundle_size:          Evaluations per worker task, shared or one per run.
        metadata:             Optional dictionary attached to every emitted result.

    Returns:
        One [`OptimizationResult`][ropt.simple.OptimizationResult] per run, in order.
    """
    check_pool(pool)
    # One pool for the whole call, even a private serial one, so that
    # concurrent runs draw their batch IDs from a single counter.
    shared_pool = pool if pool is not None else serial_pool()
    runs = broadcast_runs(config, x0, function)
    reports = broadcast_reports(report, len(runs))
    metadatas = broadcast_metadata(metadata, len(runs))
    bundle_sizes = broadcast_bundle_sizes(bundle_size, len(runs))
    jobs: list[Callable[[], OptimizationResult]] = [
        partial(
            _optimize,
            shared_pool,
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
