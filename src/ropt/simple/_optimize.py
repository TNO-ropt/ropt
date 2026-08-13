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
from ropt.exceptions import WorkflowError

from ._function import adapt_function
from ._handlers import current_handlers
from ._report import make_report_handler
from ._result import OptimizeResult
from ._session import (
    current_executor,
    current_session,
    make_task_namer,
    run_step,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any

    from numpy.typing import ArrayLike

    from ropt.components.evaluators import NameCallback
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
        handlers:             Optional local result handlers, each owned by this
                              optimization alone.
        report:               An optional callback invoked with an
                              `EvaluateResult` for each function evaluation;
                              return `True` from it to stop the optimization
                              early with `USER_ABORT`.
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
        get_name=make_task_namer(current_session(), executor),
    )


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
    get_name: NameCallback | None = None,
) -> OptimizeResult:
    context = EnOptContext.model_validate(config)
    n_obj = context.objectives.weights.size
    n_con = (
        0
        if context.nonlinear_constraints is None
        else context.nonlinear_constraints.lower_bounds.size
    )

    callback = adapt_function(function, n_obj, n_con)
    evaluator = (
        FunctionEvaluator(function=callback)
        if executor is None
        else ParallelEvaluator(function=callback, executor=executor, get_name=get_name)
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

    The runs execute concurrently on driver threads and share the session's
    worker pool, so this must be called inside a `threads`/`processes` block.
    `limit` bounds how many run simultaneously. The first run to raise cancels
    the rest and propagates (fail-fast). See
    [Running Optimizations](../running/running.md) for a walkthrough.

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
    runs = _broadcast(config, x0, function)
    reports = _broadcast_reports(report, len(runs))
    metadatas = _broadcast_metadata(metadata, len(runs))
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
            get_name=make_task_namer(session, executor),
        )
        for (run_config, run_x0, run_function), run_report, run_metadata in zip(
            runs, reports, metadatas, strict=True
        )
    ]
    return tuple(session.gather_shared(jobs, limit))


def _broadcast(
    config: dict[str, Any] | Sequence[dict[str, Any]],
    x0: ArrayLike,
    function: EvaluationFunction | Sequence[EvaluationFunction],
) -> list[tuple[dict[str, Any], ArrayLike, EvaluationFunction]]:
    configs = [config] if isinstance(config, Mapping) else list(config)
    functions = [function] if callable(function) else list(function)

    x0_array = np.asarray(x0, dtype=np.float64)
    if x0_array.ndim == 1:
        x0s: list[ArrayLike] = [x0_array]
    elif x0_array.ndim == 2:  # ruff: ignore[magic-value-comparison]
        x0s = list(x0_array)
    else:
        msg = "x0 must be a vector or a 2-D matrix of vectors."
        raise ValueError(msg)

    counts = {len(seq) for seq in (configs, functions, x0s) if len(seq) != 1}
    if len(counts) > 1:
        msg = "config, x0 and function sequences must have the same length."
        raise ValueError(msg)
    count = counts.pop() if counts else 1

    def _broadcast_seq(seq: list[Any]) -> list[Any]:
        return seq * count if len(seq) == 1 else seq

    return list(
        zip(
            _broadcast_seq(configs),
            _broadcast_seq(x0s),
            _broadcast_seq(functions),
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


def _broadcast_metadata(
    metadata: dict[str, Any] | Sequence[dict[str, Any]] | None, count: int
) -> list[dict[str, Any] | None]:
    if metadata is None:
        return [None] * count
    if isinstance(metadata, Mapping):
        return [metadata] * count
    metadatas: list[dict[str, Any] | None] = list(metadata)
    if len(metadatas) != count:
        msg = "metadata sequence length must match the number of runs."
        raise ValueError(msg)
    return metadatas
