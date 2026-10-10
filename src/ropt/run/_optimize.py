"""The optimization runs behind the session and pool methods.

One call builds a whole workflow and throws it away again: a context from the
configuration, an evaluator wired to the executor, an optimization step, and the
handlers around it. Nothing survives the call, which is what lets these
functions be called concurrently without any coordination between them.

`_optimize_many` is the same thing run several times over, on driver threads,
sharing one executor, whose workers are spread over the runs.
"""

from __future__ import annotations

from contextvars import copy_context
from functools import partial
from typing import TYPE_CHECKING, cast

import numpy as np

from ropt.components.compute_steps import OptimizationStep
from ropt.components.concurrency import AbortSignal, parent_signal, run_concurrent
from ropt.components.event_handlers import ResultsHandler
from ropt.context import EnOptContext
from ropt.enums import ExitCode
from ropt.exceptions import RunsFailedError

from ._broadcast import broadcast_arguments
from ._evaluator import make_evaluator
from ._handlers import attach_handlers
from ._result import OptimizationResult

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any

    from numpy.typing import ArrayLike

    from ropt.components.event_handlers import EventHandler
    from ropt.components.executors import Executor
    from ropt.results import FunctionResults, GradientResults

    from ._function import EvaluationFunction
    from ._report import ReportCallback
    from ._session import Session


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
    f0: FunctionResults | None,
    g0: GradientResults | None,
    report_gradients: bool,
) -> tuple[EnOptContext, OptimizationStep, ResultsHandler]:
    context = EnOptContext.model_validate(config)
    evaluator = make_evaluator(context, function, executor, bundle_size, signal)
    # This run's own handler, tracking the result the call returns; it is added
    # directly, so it stays out of the handlers the caller manages.
    result_handler = ResultsHandler(constraint_tolerance=constraint_tolerance)
    step = OptimizationStep(evaluator=evaluator, abort_signal=signal, f0=f0, g0=g0)
    step.add_event_handler(result_handler)
    attach_handlers(step, handlers, report, report_gradients=report_gradients)
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
    metadata: dict[str, Any] | None,
    optimize_many_signal: AbortSignal | None,
    f0: FunctionResults | None = None,
    g0: GradientResults | None = None,
    report_gradients: bool = False,
) -> OptimizationResult:
    if optimize_many_signal is not None and optimize_many_signal.aborting:
        # Aborted before anything is built, so an invalid config in a run that
        # never starts is not reported beside the failure that aborted it.
        return OptimizationResult(
            exit_code=optimize_many_signal.exit_code, results=None
        )
    failure_aborts_session = parent_signal() is None
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
            f0=f0,
            g0=g0,
            report_gradients=report_gradients,
        )
    except Exception:
        # No abort can reach a run being built, so this needs no exemption.
        if optimize_many_signal is not None:
            # Aborts the other runs that the same `optimize_many` started.
            optimize_many_signal.abort(ExitCode.ABORTED_ON_ERROR)
        if failure_aborts_session:
            session._fail()  # ruff: ignore[private-member-access]
        raise
    # Left outside the guard above: registration raises because the session is
    # closed, which is not this run failing.
    session._register(signal)  # ruff: ignore[private-member-access]
    parent = (
        optimize_many_signal if optimize_many_signal is not None else parent_signal()
    )
    try:
        with signal.aborts_with(parent), signal.as_parent():
            exit_code = step.run(
                context=context,
                variables=np.asarray(x0, dtype=np.float64),
                metadata=metadata,
            )
    except Exception:
        # `signal.aborting` means this run was aborted rather than failing.
        if not signal.aborting:
            # Aborts the runs and offloads started from this run's code.
            signal.abort(ExitCode.ABORTED_ON_ERROR)
            if optimize_many_signal is not None:
                # Aborts the other runs that the same `optimize_many` started.
                optimize_many_signal.abort(ExitCode.ABORTED_ON_ERROR)
            if failure_aborts_session:
                session._fail()  # ruff: ignore[private-member-access]
        raise
    finally:
        session._deregister(signal)  # ruff: ignore[private-member-access]
    results = result_handler["results"]
    if results is None or results.functions is None:
        return OptimizationResult(exit_code=exit_code, results=None)
    return OptimizationResult(
        exit_code=exit_code,
        results=results,
        gradient=result_handler["gradient"],
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
    metadata: dict[str, Any] | Sequence[dict[str, Any]] | None,
    f0: FunctionResults | Sequence[FunctionResults | None] | None = None,
    g0: GradientResults | Sequence[GradientResults | None] | None = None,
    report_gradients: bool = False,
) -> tuple[OptimizationResult, ...]:
    # Refused here rather than per run: a closed session makes the call invalid,
    # and leaving it to the runs would report it as every one of them failing.
    session._require_open()  # ruff: ignore[private-member-access]
    parent = parent_signal()
    try:
        runs, reports, metadatas, bundle_sizes, f0s, g0s = broadcast_arguments(
            config,
            x0,
            function,
            report=report,
            metadata=metadata,
            bundle_size=bundle_size,
            f0=f0,
            g0=g0,
        )
    except Exception:
        # Arguments that do not agree fail the call before any run starts.
        failure_aborts_session = parent is None
        if failure_aborts_session:
            session._fail()  # ruff: ignore[private-member-access]
        raise
    # One signal for the whole call, so an abort or a failure also reaches the
    # runs that have not started yet.
    optimize_many_signal = AbortSignal()
    session._register(optimize_many_signal)  # ruff: ignore[private-member-access]
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
            metadata=run_metadata,
            optimize_many_signal=optimize_many_signal,
            f0=run_f0,
            g0=run_g0,
            report_gradients=report_gradients,
        )
        for (
            (run_config, run_x0, run_function),
            run_report,
            run_metadata,
            run_bundle_size,
            run_f0,
            run_g0,
        ) in zip(runs, reports, metadatas, bundle_sizes, f0s, g0s, strict=True)
    ]
    # Each run starts in a copy of the caller's context, so it has the same
    # parent signal as the call itself.
    jobs = [partial(copy_context().run, job) for job in jobs]
    # Dedicated threads, not a shared thread pool: each run blocks its thread
    # while waiting for evaluations that would queue behind it in such a pool.
    try:
        with optimize_many_signal.aborts_with(parent):
            outcomes = run_concurrent(jobs, limit, interrupt=optimize_many_signal.abort)
    finally:
        session._deregister(optimize_many_signal)  # ruff: ignore[private-member-access]
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
