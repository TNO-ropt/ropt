"""The high-level ``optimize`` entry point.

One call builds a whole workflow and throws it away again: a context from the
configuration, an evaluator wired to the pool, an optimization step, and the
handlers around it. Nothing survives the call, which is what lets these
functions be called concurrently without any coordination between them.

``optimize_many`` is the same thing run several times over, on driver threads,
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
from ropt.exceptions import WorkflowError

from ._broadcast import broadcast_metadata, broadcast_reports, broadcast_runs
from ._evaluator import make_evaluator
from ._guards import check_handlers, check_pool
from ._handlers import SharedHandlers, attach_handlers, split_handlers
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


_BARE_HANDLERS = (
    "optimize_many() takes shared handlers only: its runs overlap, and a "
    "handler belongs to one run at a time. Group the handlers with "
    "`shared_handlers()` on the session, or pass `report=` for a per-run "
    "callback."
)


def optimize(  # ruff: ignore[too-many-arguments]
    config: dict[str, Any],
    x0: ArrayLike,
    function: EvaluationFunction,
    *,
    pool: WorkerPool | None = None,
    handlers: Sequence[EventHandler | SharedHandlers] | None = None,
    report: ReportCallback | None = None,
    constraint_tolerance: float = 1e-10,
    metadata: dict[str, Any] | None = None,
) -> OptimizationResult:
    """Run a single optimization.

    See [Running Optimizations](../running/running.md) for a walkthrough.

    A pool or group that is closed — because it was closed directly, or
    because its session ended — is refused here with a
    [`WorkflowError`][ropt.exceptions.WorkflowError], as is one carried into
    a worker process, where it cannot work at all.

    Without a `pool` the evaluations run in-process, on the calling thread. A
    run started from inside an evaluation needs a *different* pool: the pool it
    is already running on refuses the work.

    `handlers` mixes two kinds. An
    [`EventHandler`][ropt.components.event_handlers.EventHandler] is local: it
    is claimed for the duration of this run, and may be reused by a later run to
    accumulate results, but not shared with a concurrent one, and never
    afterwards with a [`SharedHandlers`][ropt.simple.SharedHandlers] group. A
    group is shared: this run feeds it alongside every other run that lists it.

    Returning `True` from `report` stops the optimization early with
    `USER_ABORT`. Reporting stops there, so results after it in the same batch
    are not passed on.

    Args:
        config:               The optimization configuration.
        x0:                   The initial variable vector.
        function:             The per-realization evaluation function.
        pool:                 The pool to evaluate on, from a session factory.
        handlers:             Optional local handlers and shared groups.
        report:               Optional callback invoked per function evaluation.
        constraint_tolerance: The tolerance within which a constraint is satisfied.
        metadata:             Optional dictionary attached to every emitted result.

    Returns:
        An [`OptimizationResult`][ropt.simple.OptimizationResult] describing the outcome.
    """
    check_pool(pool)
    check_handlers(handlers)
    return _optimize(
        pool if pool is not None else serial_pool(),
        config,
        x0,
        function,
        handlers=handlers,
        report=report,
        constraint_tolerance=constraint_tolerance,
        metadata=metadata,
    )


def _optimize(  # ruff: ignore[too-many-arguments]
    pool: WorkerPool,
    config: dict[str, Any],
    x0: ArrayLike,
    function: EvaluationFunction,
    *,
    handlers: Sequence[EventHandler | SharedHandlers] | None,
    report: ReportCallback | None,
    constraint_tolerance: float,
    metadata: dict[str, Any] | None = None,
) -> OptimizationResult:
    context = EnOptContext.model_validate(config)
    evaluator = make_evaluator(context, function, pool)
    # This run's own handler, tracking the result the call returns; it is added
    # directly, so it stays out of the handlers the caller manages.
    result_handler = ResultsHandler(constraint_tolerance=constraint_tolerance)
    step = OptimizationStep(evaluator=evaluator)
    step.add_event_handler(result_handler)
    with attach_handlers(step, handlers, report):
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
    handlers: Sequence[SharedHandlers] | None = None,
    report: ReportCallback | Sequence[ReportCallback] | None = None,
    limit: int | None = None,
    constraint_tolerance: float = 1e-10,
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

    Unlike `optimize`, this takes no *local* handlers: a local handler belongs
    to one run at a time and these runs overlap, so `handlers=` accepts only
    [`SharedHandlers`][ropt.simple.SharedHandlers] groups, which every run feeds
    together. `report=`, being local by nature, is the opposite: it is given per
    run, or broadcast to all of them.

    A pool or group that is closed — because it was closed directly, or
    because its session ended — is refused here with a
    [`WorkflowError`][ropt.exceptions.WorkflowError], as is one carried into
    a worker process, where it cannot work at all.

    A run started from inside an evaluation needs a *different* pool: the pool
    it is already running on refuses the work. Returning `True` from a `report`
    callback stops that run early with `USER_ABORT`. `metadata` also reaches
    each run's `function` as `context.metadata`, which makes it a way to tag a
    run, for example with `{"run_id": i}`.

    Args:
        config:               The configuration, or one per run.
        x0:                   The initial variable vector, or one per row.
        function:             The evaluation function, or one per run.
        pool:                 The pool every run evaluates on.
        handlers:             Optional shared groups, fed by every run.
        report:               Optional callback per evaluation, shared or one per run.
        limit:                The maximum number of runs to execute at once.
        constraint_tolerance: The tolerance within which a constraint is satisfied.
        metadata:             Optional dictionary attached to every emitted result.

    Returns:
        One [`OptimizationResult`][ropt.simple.OptimizationResult] per run, in order.

    Raises:
        WorkflowError: If `handlers` holds a handler that is not in a group.
    """
    check_pool(pool)
    check_handlers(handlers)
    # Rejected here rather than left to the second run's claim(), which would
    # fail mid-flight with the first run already going.
    local, groups = split_handlers(handlers)
    if local:
        raise WorkflowError(_BARE_HANDLERS)
    # One pool for the whole call, even a private serial one, so that
    # concurrent runs draw their batch IDs from a single counter.
    shared_pool = pool if pool is not None else serial_pool()
    runs = broadcast_runs(config, x0, function)
    reports = broadcast_reports(report, len(runs))
    metadatas = broadcast_metadata(metadata, len(runs))
    jobs: list[Callable[[], OptimizationResult]] = [
        partial(
            _optimize,
            shared_pool,
            run_config,
            run_x0,
            run_function,
            handlers=groups,
            report=run_report,
            constraint_tolerance=constraint_tolerance,
            metadata=run_metadata,
        )
        for (run_config, run_x0, run_function), run_report, run_metadata in zip(
            runs, reports, metadatas, strict=True
        )
    ]
    # Dedicated threads, not a shared thread pool: each run blocks its thread
    # while waiting for evaluations that would queue behind it in such a pool.
    return tuple(run_concurrent(jobs, limit))
