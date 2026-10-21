"""The optimization runs behind the module functions.

One call builds a whole workflow and throws it away again: a context from the
configuration, an evaluator wired to the executor, an optimization step, and the
handlers around it. Nothing survives the call, which is what lets these
functions be called concurrently without any coordination between them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ropt._components.compute_steps import run_optimization
from ropt._handlers import ResultsHandler
from ropt.context import EnOptContext

from ._evaluator import make_evaluator
from ._handlers import make_handlers
from ._linked_abort_scope import linked_abort_scope
from ._result import OptimizationResult

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from numpy.typing import ArrayLike

    from ropt._components.concurrency import AbortScope
    from ropt._components.executors import Executor
    from ropt._evaluation import EvaluationBatchCallback
    from ropt._handlers import EventHandler
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
    scope: AbortScope,
    report_gradients: bool,
) -> tuple[EnOptContext, EvaluationBatchCallback, ResultsHandler, list[EventHandler]]:
    context = EnOptContext.model_validate(config)
    evaluator = make_evaluator(context, function, executor, scope)
    # This run's own handler, tracking the result the call returns; it is added
    # directly, so it stays out of the handlers the caller manages.
    result_handler = ResultsHandler(constraint_tolerance=constraint_tolerance)
    ordered = [
        result_handler,
        *make_handlers(handlers, report, report_gradients=report_gradients),
    ]
    return context, evaluator, result_handler, ordered


def _optimize(  # ruff: ignore[too-many-arguments]
    session: Session | None,
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
    with linked_abort_scope(session) as scope:
        context, evaluator, result_handler, ordered = _build_optimization(
            executor,
            config,
            function,
            handlers=handlers,
            report=report,
            constraint_tolerance=constraint_tolerance,
            scope=scope,
            report_gradients=report_gradients,
        )
        with scope.as_current():
            exit_code = run_optimization(
                context=context,
                variables=np.asarray(x0, dtype=np.float64),
                evaluator=evaluator,
                handlers=ordered,
                abort_scope=scope,
                metadata=metadata,
                f0=f0,
                g0=g0,
            )
    results = result_handler["results"]
    if results is None or results.functions is None:
        return OptimizationResult(exit_code=exit_code, results=None)
    return OptimizationResult(
        exit_code=exit_code,
        results=results,
        gradient=result_handler["gradient"],
    )
