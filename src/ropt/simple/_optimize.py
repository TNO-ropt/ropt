"""The high-level ``optimize`` entry point."""

from __future__ import annotations

from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from ropt.components.compute_steps import OptimizationStep
from ropt.components.event_handlers import ResultsHandler
from ropt.exceptions import WorkflowError

from ._broadcast import broadcast_metadata, broadcast_reports, broadcast_runs
from ._evaluator import make_evaluator, run_step
from ._handlers import current_handlers
from ._report import make_report_handler
from ._result import OptimizeResult
from ._session import current_executor, current_session

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from typing import Any

    from numpy.typing import ArrayLike

    from ropt.components.event_handlers import EventHandler
    from ropt.components.executors import Executor
    from ropt.enums import ExitCode
    from ropt.results import FunctionResults

    from ._function import EvaluationFunction
    from ._handlers import HandlerScope
    from ._report import ReportCallback


def optimize(  # ruff: ignore[too-many-arguments]
    config: dict[str, Any],
    x0: ArrayLike,
    function: EvaluationFunction,
    *,
    handlers: Sequence[EventHandler] | None = None,
    report: ReportCallback | None = None,
    constraint_tolerance: float = 1e-10,
    metadata: dict[str, Any] | None = None,
) -> OptimizeResult:
    """Run a single optimization.

    See [Running Optimizations](../running/running.md) for a walkthrough.

    Args:
        config:               The optimization configuration.
        x0:                   The initial variable vector.
        function:             The per-realization evaluation function.
        handlers:             Optional local result handlers, each claimed for
                              the duration of this run. A handler may be reused
                              across sequential `optimize` calls to accumulate
                              results, but not shared with a concurrent run, and
                              not passed to
                              [`handlers`][ropt.simple.handlers] afterwards: a
                              handler used locally stays bound to its run and is
                              rejected by a shared block.
        report:               An optional callback invoked with an
                              `EvaluateResult` for each function evaluation;
                              return `True` from it to stop the optimization
                              early with `USER_ABORT`. Reporting stops there,
                              so results after it in the same batch are not
                              passed on.
        constraint_tolerance: The tolerance within which a constraint is
                              considered satisfied.
        metadata:             An optional dictionary attached to every
                              [`Results`][ropt.results.Results] this run emits,
                              for example to tag or identify the run.

    Returns:
        A [`OptimizeResult`][ropt.simple.OptimizeResult] describing the outcome.
    """
    executor = current_executor()
    return _optimize(
        executor,
        config,
        x0,
        function,
        handlers=handlers,
        report=report,
        constraint_tolerance=constraint_tolerance,
        shared=current_handlers(),
        metadata=metadata,
    )


@contextmanager
def _claim_for_run(handlers: Sequence[EventHandler]) -> Iterator[None]:
    # Release in the finally so a handler is reusable by a later run, and roll
    # back already-taken claims if a later handler in the list is claimed.
    claimed: list[EventHandler] = []
    try:
        for handler in handlers:
            handler.claim()
            claimed.append(handler)
        yield
    finally:
        for handler in claimed:
            handler.release()


def _optimize(  # ruff: ignore[too-many-arguments]
    executor: Executor | None,
    config: dict[str, Any],
    x0: ArrayLike,
    function: EvaluationFunction,
    *,
    handlers: Sequence[EventHandler] | None,
    report: ReportCallback | None,
    constraint_tolerance: float,
    shared: HandlerScope | None,
    metadata: dict[str, Any] | None = None,
) -> OptimizeResult:
    context, evaluator = make_evaluator(config, function, executor)
    result_handler = ResultsHandler(constraint_tolerance=constraint_tolerance)
    step = OptimizationStep(evaluator=evaluator)
    step.add_event_handler(result_handler)
    user_handlers = tuple(handlers or ())
    with _claim_for_run(user_handlers):
        for handler in user_handlers:
            step.add_event_handler(handler)
        if report is not None:
            step.add_event_handler(make_report_handler(report))
        if shared is not None:
            shared.attach_to(step)

        exit_code = run_step(
            step,
            context=context,
            variables=np.asarray(x0, dtype=np.float64),
            metadata=metadata,
        )
    return _build_run_result(exit_code, result_handler["results"])


def _build_run_result(
    exit_code: ExitCode, results: FunctionResults | None
) -> OptimizeResult:
    if results is None or results.functions is None:
        return OptimizeResult(
            exit_code=exit_code,
            variables=None,
            target_objective=None,
            objectives=None,
            constraints=None,
            results=None,
        )
    return OptimizeResult(
        exit_code=exit_code,
        variables=results.evaluations.variables,
        target_objective=float(results.functions.target_objective),
        objectives=results.functions.objectives,
        constraints=results.functions.constraints,
        results=results,
    )


def optimize_many(  # ruff: ignore[too-many-arguments]
    config: dict[str, Any] | Sequence[dict[str, Any]],
    x0: ArrayLike,
    function: EvaluationFunction | Sequence[EvaluationFunction],
    *,
    report: ReportCallback | Sequence[ReportCallback] | None = None,
    limit: int | None = None,
    constraint_tolerance: float = 1e-10,
    metadata: dict[str, Any] | Sequence[dict[str, Any]] | None = None,
) -> tuple[OptimizeResult, ...]:
    """Run several optimizations concurrently, sharing the open session.

    Each of `config`, `x0`, and `function` may be a single value (used for
    every run) or a sequence (one per run). Sequences set the number of runs and
    must agree in length; single values are broadcast. A single `x0` is a 1-D
    vector; a per-run sequence of `x0`s is a 2-D matrix with one vector per row.
    A sequence that is empty gives no runs, and returns no results.

    The runs execute concurrently on driver threads and share the session's
    worker pool, so this must be called inside a `threads`/`processes` block.
    `limit` bounds how many run simultaneously. See
    [Running Optimizations](../running/running.md) for a walkthrough.

    The first run to raise propagates its exception immediately (fail-fast).
    Runs that have not started are skipped, but a run already in progress
    cannot be stopped from the outside: it is abandoned, and keeps going until
    its next evaluation finds the block's executor gone. It then stops and
    returns, rather than raising — usually with
    [`ExitCode.EXECUTOR_STOPPED`][ropt.enums.ExitCode], though a run that ends
    its own optimizer loop first reports that reason instead. Either way its
    result is discarded. So leaving the block after a failure can still take as
    long as one evaluation, and with `processes` or `hpc` that is one full
    simulation.

    Unlike `optimize`, this takes no per-run `handlers`: a handler is claimed by
    one run at a time and these runs overlap. Collect results across the runs
    with a shared [`handlers`][ropt.simple.handlers] block, which tags each
    event with its run, or with a per-run `report` callback.

    Args:
        config:               The configuration, or one per run.
        x0:                   The initial variable vector, or one per row.
        function:             The evaluation function, or one per run.
        report:               An optional callback invoked with an
                              `EvaluateResult` for each function evaluation,
                              either shared by every run or one per run; return
                              `True` from it to stop that run early with
                              `USER_ABORT`.
        limit:                The maximum number of runs to execute at once.
        constraint_tolerance: The tolerance within which a constraint is
                              considered satisfied.
        metadata:             An optional dictionary attached to every
                              [`Results`][ropt.results.Results] a run emits,
                              shared by all runs or given one per run — for
                              example to tag each run with `{"run_id": i}`.

    Returns:
        One [`OptimizeResult`][ropt.simple.OptimizeResult] per run, in order.

    Raises:
        WorkflowError: If no `threads`/`processes` block is open.
    """
    session = current_session()
    if session is None:
        msg = (
            "optimize_many() requires an execution block, "
            "e.g. `with ropt.threads(...):`."
        )
        raise WorkflowError(msg)

    executor = current_executor()
    shared = current_handlers()
    runs = broadcast_runs(config, x0, function)
    reports = broadcast_reports(report, len(runs))
    metadatas = broadcast_metadata(metadata, len(runs))
    jobs: list[Callable[[], OptimizeResult]] = [
        partial(
            _optimize,
            executor,
            run_config,
            run_x0,
            run_function,
            handlers=None,
            report=run_report,
            constraint_tolerance=constraint_tolerance,
            shared=shared,
            metadata=run_metadata,
        )
        for (run_config, run_x0, run_function), run_report, run_metadata in zip(
            runs, reports, metadatas, strict=True
        )
    ]
    return tuple(session.gather_shared(jobs, limit))
