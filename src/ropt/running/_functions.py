"""The entry points: optimize, evaluate and run, with a pool or in-process.

A call is nested in the current abort scope where it is made: the scope of a
session in its `with` block, or that of the call whose code makes it. A call
given a pool is also nested in the scope of the pool's session.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

from ._evaluate import _evaluate, _evaluate_batch
from ._optimize import _optimize
from ._pool import pool_parts
from ._work import _run_work

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import ArrayLike

    from ropt.components.event_handlers import EventHandler
    from ropt.results import FunctionResults, GradientResults

    from ._function import EvaluationFunction
    from ._pool import WorkerPool
    from ._report import ReportCallback
    from ._result import EvaluationResult, OptimizationResult


def optimize(  # ruff: ignore[too-many-arguments]
    config: dict[str, Any],
    x0: ArrayLike,
    function: EvaluationFunction,
    *,
    pool: WorkerPool | None = None,
    handlers: Sequence[EventHandler] | None = None,
    report: ReportCallback | None = None,
    constraint_tolerance: float = 1e-10,
    metadata: dict[str, Any] | None = None,
    f0: FunctionResults | None = None,
    g0: GradientResults | None = None,
    report_gradients: bool = False,
) -> OptimizationResult:
    """Run a single optimization.

    The evaluations run on the workers of `pool`, or on the calling thread
    without one. See [Running Optimizations](../running/running.md) for a
    walkthrough.

    Args:
        config:               The optimization configuration.
        x0:                   The initial variable vector.
        function:             The per-realization evaluation function.
        pool:                 The pool to evaluate on, or `None` to evaluate
                              in-process.
        handlers:             Optional handlers, called in the order listed.
        report:               Optional callback invoked per evaluation.
        constraint_tolerance: The tolerance within which a constraint holds.
        metadata:             Optional dictionary attached to every result.
        f0:                   Optional function results at `x0`.
        g0:                   Optional gradient results at `x0`.
        report_gradients:     Whether `report` also receives gradient results.

    Returns:
        An [`OptimizationResult`][ropt.OptimizationResult].

    Raises:
        AbortedError:  If the optimization was aborted.
        WorkflowError: If the session of `pool` has closed.
    """  # ruff: ignore[docstring-extraneous-exception]
    session, executor = pool_parts(pool)
    return _optimize(
        session,
        executor,
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


def evaluate(  # ruff: ignore[too-many-arguments]
    config: dict[str, Any],
    variables: ArrayLike,
    function: EvaluationFunction,
    *,
    pool: WorkerPool | None = None,
    handlers: Sequence[EventHandler] | None = None,
    report: ReportCallback | None = None,
    metadata: dict[str, Any] | None = None,
) -> EvaluationResult[FunctionResults]:
    """Evaluate a single variable vector, without optimizing.

    The evaluations run on the workers of `pool`, or on the calling thread
    without one. See [Running Optimizations](../running/running.md) for a
    walkthrough.

    Args:
        config:    The optimization configuration.
        variables: The variable vector to evaluate.
        function:  The per-realization evaluation function.
        pool:      The pool to evaluate on, or `None` to evaluate in-process.
        handlers:  Optional handlers, called in the order listed.
        report:    Optional callback invoked with the results.
        metadata:  Optional dictionary attached to the results.

    Returns:
        An evaluation result.

    Raises:
        AbortedError:  If the evaluation was aborted.
        ValueError:    If `variables` is not a single vector.
        WorkflowError: If the session of `pool` has closed.
    """  # ruff: ignore[docstring-extraneous-exception]
    session, executor = pool_parts(pool)
    return _evaluate(
        session,
        executor,
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
    pool: WorkerPool | None = None,
    handlers: Sequence[EventHandler] | None = None,
    report: ReportCallback | None = None,
    metadata: dict[str, Any] | None = None,
) -> EvaluationResult[tuple[FunctionResults, ...]]:
    """Evaluate a batch of variable vectors, without optimizing.

    Each row of `variables` is one vector, and the results come back in the same
    order. The evaluations run on the workers of `pool`, or on the calling
    thread without one. See [Running Optimizations](../running/running.md) for
    a walkthrough.

    Args:
        config:    The optimization configuration.
        variables: The variable vectors to evaluate, one per row.
        function:  The per-realization evaluation function.
        pool:      The pool to evaluate on, or `None` to evaluate in-process.
        handlers:  Optional handlers, called in the order listed.
        report:    Optional callback invoked with each evaluation.
        metadata:  Optional dictionary attached to every result.

    Returns:
        An evaluation result.

    Raises:
        AbortedError:  If the evaluation was aborted.
        ValueError:    If `variables` is not a 2-D matrix.
        WorkflowError: If the session of `pool` has closed.
    """  # ruff: ignore[docstring-extraneous-exception]
    session, executor = pool_parts(pool)
    return _evaluate_batch(
        session,
        executor,
        config,
        variables,
        function,
        handlers=handlers,
        report=report,
        metadata=metadata,
    )


@overload
def run[T](work: Callable[[], T], *, pool: WorkerPool | None = None) -> T: ...


@overload
def run[T](
    work: Sequence[Callable[[], T]], *, pool: WorkerPool | None = None
) -> tuple[T, ...]: ...


def run[T](
    work: Callable[[], T] | Sequence[Callable[[], T]],
    *,
    pool: WorkerPool | None = None,
) -> T | tuple[T, ...]:
    """Run one or more arbitrary callables.

    Pass a single zero-argument callable to run one call and get its result,
    or a sequence to get a tuple of results in the order of `work`. Bind
    arguments with `functools.partial`. With a pool the callables run
    concurrently on its workers, and must be picklable for a process or HPC
    pool. Without a pool they run one after another on the calling thread. See
    [Running your own work](../running/parallel.md#offloading-your-own-work)
    for a walkthrough.

    Args:
        work: A single zero-argument callable, or a sequence of them.
        pool: The pool to run on, or `None` to run in-process.

    Returns:
        The single result, or a tuple of results in the order of `work`.

    Raises:
        AbortedError:   If an abort arrived before every callable finished.
        ExecutionError: If the machinery could not run a call.
        WorkflowError:  If the session of `pool` has closed.
    """  # ruff: ignore[docstring-extraneous-exception]
    session, executor = pool_parts(pool)
    return _run_work(session, executor, work)
