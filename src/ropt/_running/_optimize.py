"""The optimization runs behind the module functions.

One call builds a context from the configuration, an evaluator wired to the
executor, and the handlers around the optimization. Nothing survives the call,
which lets these functions be called concurrently without coordination.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ropt._handlers import ResultsHandler
from ropt.context import EnOptContext

from ._evaluator import make_evaluator
from ._handlers import make_handlers
from ._linked_abort_scope import linked_abort_scope
from ._result import OptimizationResult
from ._run_optimizer import run_optimization

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from numpy.typing import ArrayLike

    from ropt._components.executors import Executor
    from ropt._handlers import EventHandler
    from ropt.results import FunctionResults, GradientResults

    from ._function import EvaluationFunction
    from ._report import ReportCallback
    from ._session import Session


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
        context = EnOptContext.model_validate(config)
        evaluator = make_evaluator(context, function, executor, scope)
        # This run's own handler, tracking the result the call returns; it is added
        # directly, so it stays out of the handlers the caller manages.
        result_handler = ResultsHandler(constraint_tolerance=constraint_tolerance)
        ordered = [
            result_handler,
            *make_handlers(handlers, report, report_gradients=report_gradients),
        ]
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
