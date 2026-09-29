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
from ropt.components.concurrency import StopSignal
from ropt.components.event_handlers import HistoryHandler
from ropt.context import EnOptContext

from ._evaluator import make_evaluator
from ._handlers import attach_handlers

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
    metadata: dict[str, Any] | None,
) -> FunctionResults:
    array = np.asarray(variables, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(_ONE_VECTOR)
    return _run_evaluation(
        session,
        executor,
        config,
        array,
        function,
        handlers=handlers,
        report=report,
        bundle_size=bundle_size,
        metadata=metadata,
    )[0]


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
    metadata: dict[str, Any] | None,
) -> tuple[FunctionResults, ...]:
    array = np.asarray(variables, dtype=np.float64)
    if array.ndim != 2:  # ruff: ignore[magic-value-comparison]
        raise ValueError(_A_MATRIX)
    return tuple(
        _run_evaluation(
            session,
            executor,
            config,
            array,
            function,
            handlers=handlers,
            report=report,
            bundle_size=bundle_size,
            metadata=metadata,
        )
    )


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
    metadata: dict[str, Any] | None,
) -> tuple[FunctionResults, ...]:
    context = EnOptContext.model_validate(config)
    evaluator = make_evaluator(context, function, executor, bundle_size)
    # The results are collected by this run's own handler, in the order the
    # vectors were given, which is the order they are returned in.
    history = HistoryHandler()
    signal = StopSignal()
    step = EvaluationStep(evaluator=evaluator, stop_signal=signal)
    step.add_event_handler(history)
    attach_handlers(step, handlers, report)
    session._register(signal)  # ruff: ignore[private-member-access]
    try:
        step.run(
            context=context,
            variables=np.asarray(variables, dtype=np.float64),
            metadata=metadata,
        )
    finally:
        session._deregister(signal)  # ruff: ignore[private-member-access]
    return history["results"] or ()
