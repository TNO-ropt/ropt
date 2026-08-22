"""The high-level ``evaluate`` and ``evaluate_many`` entry points.

The same shape as `optimize`, with an evaluation step in place of the optimizer:
one batch of variable vectors, evaluated once, with no loop around it. The two
functions differ only in what they accept and return — one vector or a matrix of
them — so that neither has to guess which was meant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ropt.components.compute_steps import EvaluationStep
from ropt.components.event_handlers import HistoryHandler
from ropt.context import EnOptContext

from ._evaluator import make_evaluator
from ._guards import check_handlers, check_pool
from ._handlers import SharedHandlers, attach_handlers
from ._pool import serial_pool
from ._result import EvaluateResult, _build_evaluate_result

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from numpy.typing import ArrayLike

    from ropt.components.event_handlers import EventHandler
    from ropt.results import FunctionResults

    from ._function import EvaluationFunction
    from ._pool import WorkerPool
    from ._report import ReportCallback


def evaluate(  # ruff: ignore[too-many-arguments]
    config: dict[str, Any],
    variables: ArrayLike,
    function: EvaluationFunction,
    *,
    pool: WorkerPool | None = None,
    handlers: Sequence[EventHandler | SharedHandlers] | None = None,
    report: ReportCallback | None = None,
    metadata: dict[str, Any] | None = None,
) -> EvaluateResult:
    """Evaluate a single variable vector without optimizing.

    Use [`evaluate_many`][ropt.simple.evaluate_many] to evaluate several
    vectors at once. See [Running Optimizations](../running/running.md) for a
    walkthrough.

    A pool or group that is closed — because it was closed directly, or
    because its session ended — is refused here with a
    [`WorkflowError`][ropt.exceptions.WorkflowError], as is one carried into
    a worker process, where it cannot work at all.

    Args:
        config:    The optimization configuration.
        variables: The variable vector to evaluate.
        function:  The per-realization evaluation function.
        pool:      The pool to evaluate on, from a session factory such as
                   [`thread_pool`][ropt.simple.Session.thread_pool]. Without one
                   the evaluations run in-process, on the calling thread. A run
                   started from inside an evaluation needs a *different* pool:
                   the pool it is already running on refuses the work.
        handlers:  Optional result handlers, mixing local
                   [`EventHandler`][ropt.components.event_handlers.EventHandler]
                   objects with shared
                   [`SharedHandlers`][ropt.simple.SharedHandlers] groups, as
                   [`optimize`][ropt.simple.optimize] takes them.
        report:    An optional callback invoked with an `EvaluateResult` for
                   each evaluation. An evaluation is a single batch that has
                   already run by the time the callback sees it, and there is no
                   optimizer loop to interrupt, so unlike on
                   [`optimize`][ropt.simple.optimize] returning `True` cannot
                   stop anything: it only ends the reporting, and every result
                   is still returned.
        metadata:  An optional dictionary attached to the emitted
                   [`Results`][ropt.results.Results]. It also reaches
                   `function` as `context.metadata`.

    Returns:
        An [`EvaluateResult`][ropt.simple.EvaluateResult] for the vector.

    Raises:
        ValueError: If `variables` is not a single vector.
    """
    check_pool(pool)
    check_handlers(handlers)
    array = np.asarray(variables, dtype=np.float64)
    if array.ndim != 1:
        msg = "evaluate() takes a single vector; use evaluate_many() for a batch."
        raise ValueError(msg)
    results = _run_evaluation(
        pool,
        config,
        array,
        function,
        handlers=handlers,
        report=report,
        metadata=metadata,
    )
    return _build_evaluate_result(results[0])


def evaluate_many(  # ruff: ignore[too-many-arguments]
    config: dict[str, Any],
    variables: ArrayLike,
    function: EvaluationFunction,
    *,
    pool: WorkerPool | None = None,
    handlers: Sequence[EventHandler | SharedHandlers] | None = None,
    report: ReportCallback | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[EvaluateResult, ...]:
    """Evaluate a batch of variable vectors without optimizing.

    Each row of `variables` is one variable vector; the results are returned in
    the same order. See [Running Optimizations](../running/running.md) for a
    walkthrough.

    A pool or group that is closed — because it was closed directly, or
    because its session ended — is refused here with a
    [`WorkflowError`][ropt.exceptions.WorkflowError], as is one carried into
    a worker process, where it cannot work at all.

    Args:
        config:    The optimization configuration.
        variables: The variable vectors to evaluate, one per row.
        function:  The per-realization evaluation function.
        pool:      The pool to evaluate on, from a session factory such as
                   [`thread_pool`][ropt.simple.Session.thread_pool]. Without one
                   the evaluations run in-process, on the calling thread. A run
                   started from inside an evaluation needs a *different* pool:
                   the pool it is already running on refuses the work.
        handlers:  Optional result handlers, mixing local
                   [`EventHandler`][ropt.components.event_handlers.EventHandler]
                   objects with shared
                   [`SharedHandlers`][ropt.simple.SharedHandlers] groups, as
                   [`optimize`][ropt.simple.optimize] takes them.
        report:    An optional callback invoked with an `EvaluateResult` for
                   each evaluation. An evaluation is a single batch that has
                   already run by the time the callback sees it, and there is no
                   optimizer loop to interrupt, so unlike on
                   [`optimize`][ropt.simple.optimize] returning `True` cannot
                   stop anything: it only ends the reporting, and every result
                   is still returned.
        metadata:  An optional dictionary attached to every emitted
                   [`Results`][ropt.results.Results]. It also reaches
                   `function` as `context.metadata`.

    Returns:
        One [`EvaluateResult`][ropt.simple.EvaluateResult] per input vector.

    Raises:
        ValueError: If `variables` is not a 2-D matrix.
    """
    check_pool(pool)
    check_handlers(handlers)
    array = np.asarray(variables, dtype=np.float64)
    if array.ndim != 2:  # ruff: ignore[magic-value-comparison]
        msg = (
            "evaluate_many() takes a 2-D matrix of vectors (one per row); "
            "use evaluate() for a single vector."
        )
        raise ValueError(msg)
    results = _run_evaluation(
        pool,
        config,
        array,
        function,
        handlers=handlers,
        report=report,
        metadata=metadata,
    )
    return tuple(_build_evaluate_result(result) for result in results)


def _run_evaluation(  # ruff: ignore[too-many-arguments]
    pool: WorkerPool | None,
    config: dict[str, Any],
    variables: ArrayLike,
    function: EvaluationFunction,
    *,
    handlers: Sequence[EventHandler | SharedHandlers] | None,
    report: ReportCallback | None,
    metadata: dict[str, Any] | None,
) -> tuple[FunctionResults, ...]:
    context = EnOptContext.model_validate(config)
    evaluator = make_evaluator(
        context, function, pool if pool is not None else serial_pool()
    )
    # The results are collected by this run's own handler, in the order the
    # vectors were given, which is the order they are returned in.
    history = HistoryHandler()
    step = EvaluationStep(evaluator=evaluator)
    step.add_event_handler(history)
    with attach_handlers(step, handlers, report):
        step.run(
            context=context,
            variables=np.asarray(variables, dtype=np.float64),
            metadata=metadata,
        )
    return history["results"] or ()
