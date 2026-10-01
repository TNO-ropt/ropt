"""The evaluations behind the session and pool methods.

The same shape as `_optimize`, with an evaluation step in place of the
optimizer: one batch of variable vectors, evaluated once, with no loop around
it. The caller decides whether it wanted one vector or a matrix of them, so
neither has to guess which was meant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ropt.components.compute_steps import EvaluationStep
from ropt.components.concurrency import AbortSignal
from ropt.components.event_handlers import HistoryHandler
from ropt.context import EnOptContext
from ropt.enums import ExitReason

from ._evaluator import make_evaluator
from ._handlers import attach_handlers
from ._result import EvaluationResult

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from numpy.typing import ArrayLike

    from ropt.components.event_handlers import EventHandler
    from ropt.components.executors import Executor
    from ropt.results import FunctionResults

    from ._function import EvaluationFunction
    from ._report import ReportCallback
    from ._session import Session

_ONE_VECTOR = "evaluate() takes a single vector; use evaluate_batch() for a batch."

_A_MATRIX = (
    "evaluate_batch() takes a 2-D matrix of vectors (one per row); "
    "use evaluate() for a single vector."
)


def _evaluate(  # ruff: ignore[too-many-arguments]
    session: Session,
    executor: Executor | None,
    config: dict[str, Any],
    variables: ArrayLike,
    function: EvaluationFunction,
    *,
    handlers: Sequence[EventHandler] | None,
    report: ReportCallback | None,
    bundle_size: int | None,
    keep_going: bool | None,
    metadata: dict[str, Any] | None,
) -> EvaluationResult[FunctionResults | None]:
    array = np.asarray(variables, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(_ONE_VECTOR)
    outcome = _run_evaluation(
        session,
        executor,
        config,
        array,
        function,
        handlers=handlers,
        report=report,
        bundle_size=bundle_size,
        keep_going=keep_going,
        metadata=metadata,
    )
    return EvaluationResult(
        exit_reason=outcome.exit_reason,
        results=outcome.results[0] if outcome.results else None,
    )


def _evaluate_batch(  # ruff: ignore[too-many-arguments]
    session: Session,
    executor: Executor | None,
    config: dict[str, Any],
    variables: ArrayLike,
    function: EvaluationFunction,
    *,
    handlers: Sequence[EventHandler] | None,
    report: ReportCallback | None,
    bundle_size: int | None,
    keep_going: bool | None,
    metadata: dict[str, Any] | None,
) -> EvaluationResult[tuple[FunctionResults, ...]]:
    array = np.asarray(variables, dtype=np.float64)
    if array.ndim != 2:  # ruff: ignore[magic-value-comparison]
        raise ValueError(_A_MATRIX)
    return _run_evaluation(
        session,
        executor,
        config,
        array,
        function,
        handlers=handlers,
        report=report,
        bundle_size=bundle_size,
        keep_going=keep_going,
        metadata=metadata,
    )


def _build_evaluation(  # ruff: ignore[too-many-arguments]
    executor: Executor | None,
    config: dict[str, Any],
    function: EvaluationFunction,
    *,
    handlers: Sequence[EventHandler] | None,
    report: ReportCallback | None,
    bundle_size: int | None,
    signal: AbortSignal,
) -> tuple[EnOptContext, EvaluationStep, HistoryHandler]:
    context = EnOptContext.model_validate(config)
    evaluator = make_evaluator(context, function, executor, bundle_size, signal)
    # The results are collected by this run's own handler, in the order the
    # vectors were given, which is the order they are returned in.
    history = HistoryHandler()
    step = EvaluationStep(evaluator=evaluator, abort_signal=signal)
    step.add_event_handler(history)
    attach_handlers(step, handlers, report)
    return context, step, history


def _run_evaluation(  # ruff: ignore[too-many-arguments]
    session: Session,
    executor: Executor | None,
    config: dict[str, Any],
    variables: ArrayLike,
    function: EvaluationFunction,
    *,
    handlers: Sequence[EventHandler] | None,
    report: ReportCallback | None,
    bundle_size: int | None,
    keep_going: bool | None,
    metadata: dict[str, Any] | None,
) -> EvaluationResult[tuple[FunctionResults, ...]]:
    signal = AbortSignal()
    try:
        context, step, history = _build_evaluation(
            executor,
            config,
            function,
            handlers=handlers,
            report=report,
            bundle_size=bundle_size,
            signal=signal,
        )
    except Exception:
        # No abort can reach a run being built, so this needs no exemption.
        session._fail()  # ruff: ignore[private-member-access]
        raise
    # Left outside the guard above: a refusal here means the session is
    # closing, not that this run failed.
    session._register(  # ruff: ignore[private-member-access]
        signal,
        keep_going=session._resolve_keep_going(keep_going=keep_going),  # ruff: ignore[private-member-access]
    )
    try:
        step.run(
            context=context,
            variables=np.asarray(variables, dtype=np.float64),
            metadata=metadata,
        )
    except Exception:
        # Being cut off is not a failure, so it must not cut off anything else.
        if not signal.aborting:
            session._fail()  # ruff: ignore[private-member-access]
        raise
    finally:
        session._deregister(signal)  # ruff: ignore[private-member-access]
    results = tuple(history["results"] or ())
    # An abort that arrived too late to cost the batch anything did not abort
    # it: the step reports its results either way.
    if signal.aborting and not results:
        return EvaluationResult(exit_reason=signal.exit_reason, results=())
    return EvaluationResult(exit_reason=ExitReason.FINISHED, results=results)
