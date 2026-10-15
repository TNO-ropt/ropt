"""The optimization runs behind the session and pool methods.

One call builds a whole workflow and throws it away again: a context from the
configuration, an evaluator wired to the executor, an optimization step, and the
handlers around it. Nothing survives the call, which is what lets these
functions be called concurrently without any coordination between them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ropt.components.compute_steps import OptimizationStep
from ropt.components.concurrency import AbortSignal, parent_signal
from ropt.components.event_handlers import ResultsHandler
from ropt.context import EnOptContext
from ropt.enums import ExitCode
from ropt.exceptions import AbortedError, ExecutorStopped

from ._evaluator import make_evaluator
from ._handlers import attach_handlers
from ._result import OptimizationResult

if TYPE_CHECKING:
    from collections.abc import Sequence
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
    signal: AbortSignal,
    f0: FunctionResults | None,
    g0: GradientResults | None,
    report_gradients: bool,
) -> tuple[EnOptContext, OptimizationStep, ResultsHandler]:
    context = EnOptContext.model_validate(config)
    evaluator = make_evaluator(context, function, executor, signal)
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
    metadata: dict[str, Any] | None,
    f0: FunctionResults | None = None,
    g0: GradientResults | None = None,
    report_gradients: bool = False,
) -> OptimizationResult:
    parent = parent_signal()
    failure_aborts_session = parent is None
    signal = AbortSignal()
    try:
        context, step, result_handler = _build_optimization(
            executor,
            config,
            function,
            handlers=handlers,
            report=report,
            constraint_tolerance=constraint_tolerance,
            signal=signal,
            f0=f0,
            g0=g0,
            report_gradients=report_gradients,
        )
    except Exception:
        # No abort can reach a run being built, so this needs no exemption.
        if failure_aborts_session:
            session._fail()  # ruff: ignore[private-member-access]
        raise
    # Left outside the guard above: registration raises because the session is
    # closed, which is not this run failing.
    session._register(signal)  # ruff: ignore[private-member-access]
    try:
        with signal.aborts_with(parent), signal.as_parent():
            exit_code = step.run(
                context=context,
                variables=np.asarray(x0, dtype=np.float64),
                metadata=metadata,
            )
    except ExecutorStopped:
        # Not a failure of this run, so it is reported the way an abort is
        # rather than escaping.
        raise AbortedError(ExitCode.EXECUTOR_SHUT_DOWN) from None
    except Exception:
        # `signal.aborting` means this run was aborted rather than failing.
        if not signal.aborting:
            # Aborts the tasks started from this optimization's code.
            signal.abort(ExitCode.ABORTED_ON_ERROR)
            if failure_aborts_session:
                session._fail()  # ruff: ignore[private-member-access]
        raise
    except BaseException:
        # Not a failure, but the work started from this run must not outlive it.
        signal.abort(ExitCode.ABORTED)
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
