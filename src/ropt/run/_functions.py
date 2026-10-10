"""The entry points for a run that belongs to no session.

Each opens a session of its own for the length of the call, so a run always has
one. Nothing else can reach that session, which is exactly what "outside a
session" means: no `Session.abort` reaches the run directly, and its failure
aborts only what it started. Started from inside another run or offload, it is
still nested there and is aborted with it.

Give a run a [`session`][ropt.session], or one of its pools, when it
should be part of something larger.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._session import session

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike

    from ropt.components.event_handlers import EventHandler
    from ropt.results import FunctionResults, GradientResults

    from ._function import EvaluationFunction
    from ._report import ReportCallback
    from ._result import EvaluationResult, OptimizationResult


def optimize(  # ruff: ignore[too-many-arguments]
    config: dict[str, Any],
    x0: ArrayLike,
    function: EvaluationFunction,
    *,
    handlers: Sequence[EventHandler] | None = None,
    report: ReportCallback | None = None,
    constraint_tolerance: float = 1e-10,
    metadata: dict[str, Any] | None = None,
    f0: FunctionResults | None = None,
    g0: GradientResults | None = None,
    report_gradients: bool = False,
) -> OptimizationResult:
    """Run a single optimization in-process.

    The evaluations run on the calling thread. See
    [Running Optimizations](../running/running.md) for a walkthrough, and
    [`Session.optimize`][ropt.Session.optimize] for the same run on a
    session you hold.

    Args:
        config:               The optimization configuration.
        x0:                   The initial variable vector.
        function:             The per-realization evaluation function.
        handlers:             Optional handlers, called in the order listed.
        report:               Optional callback invoked per evaluation.
        constraint_tolerance: The tolerance within which a constraint holds.
        metadata:             Optional dictionary attached to every result.
        f0:                   Optional function results at `x0`.
        g0:                   Optional gradient results at `x0`.
        report_gradients:     Whether `report` also receives gradient results.

    Returns:
        An [`OptimizationResult`][ropt.OptimizationResult].
    """
    with session() as opened:
        return opened.optimize(
            config,
            x0,
            function,
            handlers=handlers,
            report=report,
            constraint_tolerance=constraint_tolerance,
            metadata=metadata,
            f0=f0,
            g0=g0,
            report_gradients=report_gradients,
        )


def optimize_many(  # ruff: ignore[too-many-arguments]
    config: dict[str, Any] | Sequence[dict[str, Any]],
    x0: ArrayLike,
    function: EvaluationFunction | Sequence[EvaluationFunction],
    *,
    handlers: Sequence[EventHandler] | None = None,
    report: ReportCallback | Sequence[ReportCallback] | None = None,
    limit: int | None = None,
    constraint_tolerance: float = 1e-10,
    metadata: dict[str, Any] | Sequence[dict[str, Any]] | None = None,
    f0: FunctionResults | Sequence[FunctionResults | None] | None = None,
    g0: GradientResults | Sequence[GradientResults | None] | None = None,
    report_gradients: bool = False,
) -> tuple[OptimizationResult, ...]:
    """Run several optimizations concurrently, in-process.

    The runs overlap on driver threads, but each evaluates on its own thread, so
    `function` is called by several threads at once and must tolerate that. See
    [Evaluating in Parallel](../running/parallel.md) for a
    walkthrough, and
    [`WorkerPool.optimize_many`][ropt.WorkerPool.optimize_many] to give
    the evaluations workers.

    Args:
        config:               The configuration, or one per run.
        x0:                   The initial vector, or one per row.
        function:             The evaluation function, or one per run.
        handlers:             Optional handlers, fed by every run.
        report:               Optional callback, shared or one per run.
        limit:                The maximum number of runs at once.
        constraint_tolerance: The tolerance within which a constraint holds.
        metadata:             Optional dictionary attached to every result.
        f0:                   Optional function results at `x0`, shared or one
                              per run.
        g0:                   Optional gradient results at `x0`, shared or one
                              per run.
        report_gradients:     Whether `report` also receives gradient results.

    Returns:
        One [`OptimizationResult`][ropt.OptimizationResult] per run.

    Raises:
        RunsFailedError: If any of the runs raised.
        ValueError:      If `x0` has the wrong shape, or the sequences given
                         per run disagree in length.
    """  # ruff: ignore[docstring-extraneous-exception]
    with session() as opened:
        return opened.optimize_many(
            config,
            x0,
            function,
            handlers=handlers,
            report=report,
            limit=limit,
            constraint_tolerance=constraint_tolerance,
            metadata=metadata,
            f0=f0,
            g0=g0,
            report_gradients=report_gradients,
        )


def evaluate(  # ruff: ignore[too-many-arguments]
    config: dict[str, Any],
    variables: ArrayLike,
    function: EvaluationFunction,
    *,
    handlers: Sequence[EventHandler] | None = None,
    report: ReportCallback | None = None,
    metadata: dict[str, Any] | None = None,
) -> EvaluationResult[FunctionResults | None]:
    """Evaluate a single variable vector in-process, without optimizing.

    See [Running Optimizations](../running/running.md) for a walkthrough.

    Args:
        config:    The optimization configuration.
        variables: The variable vector to evaluate.
        function:  The per-realization evaluation function.
        handlers:  Optional handlers, called in the order listed.
        report:    Optional callback invoked with the results.
        metadata:  Optional dictionary attached to the results.

    Returns:
        An [`EvaluationResult`][ropt.EvaluationResult] whose `results` is
        the [`FunctionResults`][ropt.results.FunctionResults] for the vector.

    Raises:
        ValueError: If `variables` is not a single vector.
    """  # ruff: ignore[docstring-extraneous-exception]
    with session() as opened:
        return opened.evaluate(
            config,
            variables,
            function,
            handlers=handlers,
            report=report,
            metadata=metadata,
        )


def evaluate_batch(  # ruff: ignore[too-many-arguments]
    config: dict[str, Any],
    variables: ArrayLike,
    function: EvaluationFunction,
    *,
    handlers: Sequence[EventHandler] | None = None,
    report: ReportCallback | None = None,
    metadata: dict[str, Any] | None = None,
) -> EvaluationResult[tuple[FunctionResults, ...]]:
    """Evaluate a batch of variable vectors in-process, without optimizing.

    Each row of `variables` is one vector, and the results come back in the same
    order. See [Running Optimizations](../running/running.md) for a walkthrough.

    Args:
        config:    The optimization configuration.
        variables: The variable vectors to evaluate, one per row.
        function:  The per-realization evaluation function.
        handlers:  Optional handlers, called in the order listed.
        report:    Optional callback invoked with each evaluation.
        metadata:  Optional dictionary attached to every result.

    Returns:
        An [`EvaluationResult`][ropt.EvaluationResult] whose `results`
        holds one [`FunctionResults`][ropt.results.FunctionResults] per vector.

    Raises:
        ValueError: If `variables` is not a 2-D matrix.
    """  # ruff: ignore[docstring-extraneous-exception]
    with session() as opened:
        return opened.evaluate_batch(
            config,
            variables,
            function,
            handlers=handlers,
            report=report,
            metadata=metadata,
        )
