"""The high-level ``optimize`` entry point."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from ropt.components.compute_steps import OptimizationStep
from ropt.components.evaluators import FunctionEvaluator, ParallelEvaluator
from ropt.components.event_handlers import ResultsHandler
from ropt.context import EnOptContext

from ._handlers import current_handlers
from ._objective import adapt_objective
from ._report import make_report_handler
from ._result import OptimizeResult
from ._session import current_executor, current_session, run_step

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any

    from numpy.typing import ArrayLike

    from ropt.components.event_handlers import EventHandler
    from ropt.components.executors import Executor
    from ropt.enums import ExitCode
    from ropt.results import FunctionResults

    from ._handlers import HandlerScope
    from ._objective import ObjectiveCallback
    from ._report import ReportCallback


def optimize(  # ruff: ignore[too-many-arguments]
    config: dict[str, Any],
    x0: ArrayLike,
    objective: ObjectiveCallback,
    *,
    handlers: Sequence[EventHandler] | None = None,
    report: ReportCallback | None = None,
    constraint_tolerance: float = 1e-10,
) -> OptimizeResult:
    """Run a single optimization.

    See [High-Level API](../usage/simple.md) for a walkthrough.

    Args:
        config:               The optimization configuration.
        x0:                   The initial variable vector.
        objective:            The per-realization objective callback.
        handlers:             Optional local result handlers, each owned by this
                              optimization alone.
        report:               An optional callback invoked with an
                              `EvaluateResult` for each function evaluation.
        constraint_tolerance: The tolerance within which a constraint is
                              considered satisfied.

    Returns:
        A [`OptimizeResult`][ropt.simple.OptimizeResult] describing the outcome.
    """
    return _optimize(
        current_executor(),
        config,
        x0,
        objective,
        handlers=handlers,
        report=report,
        constraint_tolerance=constraint_tolerance,
        shared=current_handlers(),
    )


def _optimize(  # ruff: ignore[too-many-arguments]
    executor: Executor | None,
    config: dict[str, Any],
    x0: ArrayLike,
    objective: ObjectiveCallback,
    *,
    handlers: Sequence[EventHandler] | None,
    report: ReportCallback | None,
    constraint_tolerance: float,
    shared: HandlerScope | None,
) -> OptimizeResult:
    context = EnOptContext.model_validate(config)
    n_obj = context.objectives.weights.size
    n_con = (
        0
        if context.nonlinear_constraints is None
        else context.nonlinear_constraints.lower_bounds.size
    )

    callback = adapt_objective(objective, n_obj, n_con)
    evaluator = (
        FunctionEvaluator(function=callback)
        if executor is None
        else ParallelEvaluator(function=callback, executor=executor)
    )
    result_handler = ResultsHandler(constraint_tolerance=constraint_tolerance)
    step = OptimizationStep(evaluator=evaluator)
    step.add_event_handler(result_handler)
    for handler in handlers or ():
        handler.claim()
        step.add_event_handler(handler)
    if report is not None:
        step.add_event_handler(make_report_handler(report))
    if shared is not None:
        shared.attach_to(step)

    exit_code = run_step(
        step, context=context, variables=np.asarray(x0, dtype=np.float64)
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
    objective: ObjectiveCallback | Sequence[ObjectiveCallback],
    *,
    report: ReportCallback | Sequence[ReportCallback] | None = None,
    limit: int | None = None,
    constraint_tolerance: float = 1e-10,
) -> tuple[OptimizeResult, ...]:
    """Run several optimizations concurrently, sharing the open session.

    Each of `config`, `x0`, and `objective` may be a single value (used for
    every run) or a sequence (one per run). Sequences set the number of runs and
    must agree in length; single values are broadcast. A single `x0` is a 1-D
    vector; a per-run sequence of `x0`s is a 2-D matrix with one vector per row.

    The runs execute concurrently on driver threads and share the session's
    worker pool, so this must be called inside a `threads`/`processes` block.
    `limit` bounds how many run simultaneously. The first run to raise cancels
    the rest and propagates (fail-fast). See
    [High-Level API](../usage/simple.md) for a walkthrough.

    Args:
        config:               The configuration, or one per run.
        x0:                   The initial variable vector, or one per row.
        objective:            The objective callback, or one per run.
        report:               An optional callback invoked with an
                              `EvaluateResult` for each function evaluation,
                              either shared by every run or one per run.
        limit:                The maximum number of runs to execute at once.
        constraint_tolerance: The tolerance within which a constraint is
                              considered satisfied.

    Returns:
        One [`OptimizeResult`][ropt.simple.OptimizeResult] per run, in order.

    Raises:
        RuntimeError: If no `threads`/`processes` block is open.
    """
    session = current_session()
    if session is None:
        msg = (
            "optimize_many() requires an execution block, "
            "e.g. `with ropt.threads(...):`."
        )
        raise RuntimeError(msg)

    executor = current_executor()
    shared = current_handlers()
    runs = _broadcast(config, x0, objective)
    reports = _broadcast_reports(report, len(runs))
    jobs: list[Callable[[], OptimizeResult]] = [
        partial(
            _optimize,
            executor,
            run_config,
            run_x0,
            run_objective,
            handlers=None,
            report=run_report,
            constraint_tolerance=constraint_tolerance,
            shared=shared,
        )
        for (run_config, run_x0, run_objective), run_report in zip(
            runs, reports, strict=True
        )
    ]
    return tuple(session.gather(jobs, limit))


def _broadcast(
    config: dict[str, Any] | Sequence[dict[str, Any]],
    x0: ArrayLike,
    objective: ObjectiveCallback | Sequence[ObjectiveCallback],
) -> list[tuple[dict[str, Any], ArrayLike, ObjectiveCallback]]:
    configs = [config] if isinstance(config, Mapping) else list(config)
    objectives = [objective] if callable(objective) else list(objective)

    x0_array = np.asarray(x0, dtype=np.float64)
    if x0_array.ndim == 1:
        x0s: list[ArrayLike] = [x0_array]
    elif x0_array.ndim == 2:  # ruff: ignore[magic-value-comparison]
        x0s = list(x0_array)
    else:
        msg = "x0 must be a vector or a 2-D matrix of vectors."
        raise ValueError(msg)

    counts = {len(seq) for seq in (configs, objectives, x0s) if len(seq) != 1}
    if len(counts) > 1:
        msg = "config, x0 and objective sequences must have the same length."
        raise ValueError(msg)
    count = counts.pop() if counts else 1

    def _broadcast_seq(seq: list[Any]) -> list[Any]:
        return seq * count if len(seq) == 1 else seq

    return list(
        zip(
            _broadcast_seq(configs),
            _broadcast_seq(x0s),
            _broadcast_seq(objectives),
            strict=True,
        )
    )


def _broadcast_reports(
    report: ReportCallback | Sequence[ReportCallback] | None, count: int
) -> list[ReportCallback | None]:
    if report is None:
        return [None] * count
    if callable(report):
        return [report] * count
    reports: list[ReportCallback | None] = list(report)
    if len(reports) != count:
        msg = "report sequence length must match the number of runs."
        raise ValueError(msg)
    return reports
