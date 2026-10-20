"""The evaluations behind the module functions.

The same shape as `_optimize`, with an evaluation step in place of the
optimizer: one batch of variable vectors, evaluated once, with no loop around
it. The caller decides whether it wanted one vector or a matrix of them, so
neither has to guess which was meant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ropt._components.compute_steps import EvaluationStep
from ropt._handlers import HistoryHandler
from ropt.context import EnOptContext

from ._evaluator import make_evaluator
from ._handlers import attach_handlers
from ._linked_abort_scope import linked_abort_scope
from ._result import EvaluationResult

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from numpy.typing import ArrayLike, NDArray

    from ropt._components.concurrency import AbortScope
    from ropt._components.executors import Executor
    from ropt._handlers import EventHandler
    from ropt.results import FunctionResults

    from ._function import EvaluationFunction
    from ._report import ReportCallback
    from ._session import Session

_ONE_VECTOR = "evaluate() takes a single vector; use evaluate_batch() for a batch."

_A_MATRIX = (
    "evaluate_batch() takes a 2-D matrix of vectors (one per row); "
    "use evaluate() for a single vector."
)


def _as_vector(variables: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(variables, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(_ONE_VECTOR)
    return array


def _as_matrix(variables: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(variables, dtype=np.float64)
    if array.ndim != 2:  # ruff: ignore[magic-value-comparison]
        raise ValueError(_A_MATRIX)
    return array


def _evaluate(  # ruff: ignore[too-many-arguments]
    session: Session | None,
    executor: Executor | None,
    config: dict[str, Any],
    variables: ArrayLike,
    function: EvaluationFunction,
    *,
    handlers: Sequence[EventHandler] | None,
    report: ReportCallback | None,
    metadata: dict[str, Any] | None,
) -> EvaluationResult[FunctionResults]:
    array = _as_vector(variables)
    outcome = _run_evaluation(
        session,
        executor,
        config,
        array,
        function,
        handlers=handlers,
        report=report,
        metadata=metadata,
    )
    return EvaluationResult(results=outcome.results[0])


def _evaluate_batch(  # ruff: ignore[too-many-arguments]
    session: Session | None,
    executor: Executor | None,
    config: dict[str, Any],
    variables: ArrayLike,
    function: EvaluationFunction,
    *,
    handlers: Sequence[EventHandler] | None,
    report: ReportCallback | None,
    metadata: dict[str, Any] | None,
) -> EvaluationResult[tuple[FunctionResults, ...]]:
    array = _as_matrix(variables)
    return _run_evaluation(
        session,
        executor,
        config,
        array,
        function,
        handlers=handlers,
        report=report,
        metadata=metadata,
    )


def _build_evaluation(  # ruff: ignore[too-many-arguments]
    executor: Executor | None,
    config: dict[str, Any],
    function: EvaluationFunction,
    *,
    handlers: Sequence[EventHandler] | None,
    report: ReportCallback | None,
    scope: AbortScope,
) -> tuple[EnOptContext, EvaluationStep, HistoryHandler]:
    context = EnOptContext.model_validate(config)
    evaluator = make_evaluator(context, function, executor, scope)
    # The results are collected by this run's own handler, in the order the
    # vectors were given, which is the order they are returned in.
    history = HistoryHandler()
    step = EvaluationStep(evaluator=evaluator, abort_scope=scope)
    step.add_event_handler(history)
    attach_handlers(step, handlers, report)
    return context, step, history


def _run_evaluation(  # ruff: ignore[too-many-arguments]
    session: Session | None,
    executor: Executor | None,
    config: dict[str, Any],
    variables: ArrayLike,
    function: EvaluationFunction,
    *,
    handlers: Sequence[EventHandler] | None,
    report: ReportCallback | None,
    metadata: dict[str, Any] | None,
) -> EvaluationResult[tuple[FunctionResults, ...]]:
    with linked_abort_scope(session) as scope:
        context, step, history = _build_evaluation(
            executor,
            config,
            function,
            handlers=handlers,
            report=report,
            scope=scope,
        )
        with scope.as_current():
            step.run(
                context=context,
                variables=np.asarray(variables, dtype=np.float64),
                metadata=metadata,
            )
    return EvaluationResult(results=tuple(history["results"] or ()))
