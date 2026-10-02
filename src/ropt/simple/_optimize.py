"""The optimization runs behind the session and pool methods.

One call builds a whole workflow and throws it away again: a context from the
configuration, an evaluator wired to the executor, an optimization step, and the
handlers around it. Nothing survives the call, which is what lets these
functions be called concurrently without any coordination between them.

`_optimize_many` is the same thing run several times over, on driver threads,
sharing one executor, whose workers are spread over the runs.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, cast

import numpy as np

from ropt.components.compute_steps import OptimizationStep
from ropt.components.concurrency import AbortSignal, run_concurrent
from ropt.components.event_handlers import ResultsHandler
from ropt.context import EnOptContext
from ropt.exceptions import RunsFailedError

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
    from ._session import Session


def _cut_off(signal: AbortSignal, parent_signal: AbortSignal) -> None:
    # Registered before the parent aborts, so the reason is read when it fires.
    signal.abort(parent_signal.exit_code)


def _build_optimization(  # ruff: ignore[too-many-arguments]
    executor: Executor | None,
    config: dict[str, Any],
    function: EvaluationFunction,
    *,
    handlers: Sequence[EventHandler] | None,
    report: ReportCallback | None,
    constraint_tolerance: float,
    bundle_size: int | None,
    signal: AbortSignal,
) -> tuple[EnOptContext, OptimizationStep, ResultsHandler]:
    context = EnOptContext.model_validate(config)
    evaluator = make_evaluator(context, function, executor, bundle_size, signal)
    # This run's own handler, tracking the result the call returns; it is added
    # directly, so it stays out of the handlers the caller manages.
    result_handler = ResultsHandler(constraint_tolerance=constraint_tolerance)
    step = OptimizationStep(evaluator=evaluator, abort_signal=signal)
    step.add_event_handler(result_handler)
    attach_handlers(step, handlers, report)
    return context, step, result_handler


def _optimize(  # ruff: ignore[too-many-arguments]
    session: Session,
    executor: Executor | None,
    config: dict[str, Any],
    x0: ArrayLike,
    function: EvaluationFunction,
    *,
    handlers: Sequence[EventHandler] | None,
    report: ReportCallback | None,
    constraint_tolerance: float,
    bundle_size: int | None,
    keep_going: bool | None,
    metadata: dict[str, Any] | None,
    parent_signal: AbortSignal | None,
) -> OptimizationResult:
    if parent_signal is not None and parent_signal.aborting:
        # Cut off before anything is built, so an invalid config in a run that
        # never starts is not reported beside the failure that stopped it.
        return OptimizationResult(exit_code=parent_signal.exit_code, results=None)
    signal = AbortSignal()
    try:
        context, step, result_handler = _build_optimization(
            executor,
            config,
            function,
            handlers=handlers,
            report=report,
            constraint_tolerance=constraint_tolerance,
            bundle_size=bundle_size,
            signal=signal,
        )
    except Exception:
        # No abort can reach a run being built, so this needs no exemption.
        session._fail()  # ruff: ignore[private-member-access]
        raise
    # Left outside the guard above: registration raises because the session is
    # closed, which is not this run failing.
    session._register(  # ruff: ignore[private-member-access]
        signal,
        keep_going=session._resolve_keep_going(keep_going=keep_going),  # ruff: ignore[private-member-access]
    )
    cut_off = None
    if parent_signal is not None:
        # Runs at once when the parent is already aborting, which is what cuts
        # off a run that was still queued when the abort arrived.
        cut_off = partial(_cut_off, signal, parent_signal)
        parent_signal.add_callback(cut_off)
    try:
        exit_code = step.run(
            context=context,
            variables=np.asarray(x0, dtype=np.float64),
            metadata=metadata,
        )
    except Exception:
        # `signal.aborting` means this run was cut off rather than failing, so
        # `_fail` is skipped and the other runs are left alone.
        if not signal.aborting:
            session._fail()  # ruff: ignore[private-member-access]
        raise
    finally:
        session._deregister(signal)  # ruff: ignore[private-member-access]
        if parent_signal is not None and cut_off is not None:
            parent_signal.remove_callback(cut_off)
    results = result_handler["results"]
    return OptimizationResult(
        exit_code=exit_code,
        results=None if results is None or results.functions is None else results,
    )


def _optimize_many(  # ruff: ignore[too-many-arguments]
    session: Session,
    executor: Executor | None,
    config: dict[str, Any] | Sequence[dict[str, Any]],
    x0: ArrayLike,
    function: EvaluationFunction | Sequence[EvaluationFunction],
    *,
    handlers: Sequence[EventHandler] | None,
    report: ReportCallback | Sequence[ReportCallback] | None,
    limit: int | None,
    constraint_tolerance: float,
    bundle_size: int | Sequence[int | None] | None,
    keep_going: bool | None,
    metadata: dict[str, Any] | Sequence[dict[str, Any]] | None,
) -> tuple[OptimizationResult, ...]:
    # Refused here rather than per run: a closed session makes the call invalid,
    # and leaving it to the runs would report it as every one of them failing.
    session._require_open()  # ruff: ignore[private-member-access]
    try:
        runs = broadcast_runs(config, x0, function)
        reports = broadcast_reports(report, len(runs))
        metadatas = broadcast_metadata(metadata, len(runs))
        bundle_sizes = broadcast_bundle_sizes(bundle_size, len(runs))
    except Exception:
        # Arguments that do not agree are a failed call, and stop the rest of
        # the session as a failed run does.
        session._fail()  # ruff: ignore[private-member-access]
        raise
    # One signal for the whole call, so an abort reaches the runs that have not
    # started yet. It carries the call's own `keep_going`, so a failure cuts off
    # a run still queued behind `limit` as it cuts off one already running.
    parent_signal = AbortSignal()
    session._register(  # ruff: ignore[private-member-access]
        parent_signal,
        keep_going=session._resolve_keep_going(keep_going=keep_going),  # ruff: ignore[private-member-access]
    )
    jobs: list[Callable[[], OptimizationResult]] = [
        partial(
            _optimize,
            session,
            executor,
            run_config,
            run_x0,
            run_function,
            handlers=handlers,
            report=run_report,
            constraint_tolerance=constraint_tolerance,
            bundle_size=run_bundle_size,
            keep_going=keep_going,
            metadata=run_metadata,
            parent_signal=parent_signal,
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
    try:
        outcomes = run_concurrent(jobs, limit, interrupt=parent_signal.abort)
    finally:
        session._deregister(parent_signal)  # ruff: ignore[private-member-access]
    for outcome in outcomes:
        # A KeyboardInterrupt or SystemExit means the program is ending, so it
        # is re-raised rather than collected into `RunsFailedError`.
        if isinstance(outcome, BaseException) and not isinstance(outcome, Exception):
            raise outcome
    errors = [outcome for outcome in outcomes if isinstance(outcome, Exception)]
    if errors:
        raise RunsFailedError(
            cast("list[OptimizationResult | Exception]", outcomes)
        ) from errors[0]
    return cast("tuple[OptimizationResult, ...]", tuple(outcomes))
