"""The high-level `evaluate` and `evaluate_many` entry points.

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


def evaluate(  # ruff: ignore[too-many-arguments]
    config: dict[str, Any],
    variables: ArrayLike,
    function: EvaluationFunction,
    *,
    executor: Executor | None = None,
    handlers: Sequence[EventHandler] | None = None,
    report: ReportCallback | None = None,
    bundle_size: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> FunctionResults:
    """Evaluate a single variable vector without optimizing.

    Use [`evaluate_many`][ropt.simple.evaluate_many] to evaluate several
    vectors at once. See [Running Optimizations](../running/running.md) for a
    walkthrough.

    Without an `executor` the evaluations run in-process, on the calling thread,
    and `bundle_size` does not apply. A closed executor raises a
    [`WorkflowError`][ropt.exceptions.WorkflowError] at the first evaluation. A
    run started from inside an evaluation needs an executor with workers of its
    own: the one it is already running on refuses the work. `handlers` takes
    [`EventHandler`][ropt.components.event_handlers.EventHandler] objects, as
    [`optimize`][ropt.simple.optimize] does.

    An evaluation is a single batch that has already run by the time `report`
    sees it, and there is no optimizer loop to interrupt, so unlike on
    [`optimize`][ropt.simple.optimize] returning `True` cannot stop anything: it
    only ends the reporting, and every result is still returned. `metadata` also
    reaches `function` as `context.metadata`.

    Args:
        config:      The optimization configuration.
        variables:   The variable vector to evaluate.
        function:    The per-realization evaluation function.
        executor:    The executor to evaluate on, or `None`.
        handlers:    Optional handlers, called in the order listed.
        report:      Optional callback invoked with each evaluation's results.
        bundle_size: Evaluations per worker task, `None` for the executor's own.
        metadata:    Optional dictionary attached to the emitted results.

    Returns:
        The [`FunctionResults`][ropt.results.FunctionResults] for the vector.

    Raises:
        ValueError: If `variables` is not a single vector.
    """
    array = np.asarray(variables, dtype=np.float64)
    if array.ndim != 1:
        msg = "evaluate() takes a single vector; use evaluate_many() for a batch."
        raise ValueError(msg)
    results = _run_evaluation(
        executor,
        config,
        array,
        function,
        handlers=handlers,
        report=report,
        bundle_size=bundle_size,
        metadata=metadata,
    )
    return results[0]


def evaluate_many(  # ruff: ignore[too-many-arguments]
    config: dict[str, Any],
    variables: ArrayLike,
    function: EvaluationFunction,
    *,
    executor: Executor | None = None,
    handlers: Sequence[EventHandler] | None = None,
    report: ReportCallback | None = None,
    bundle_size: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[FunctionResults, ...]:
    """Evaluate a batch of variable vectors without optimizing.

    Each row of `variables` is one variable vector; the results are returned in
    the same order. See [Running Optimizations](../running/running.md) for a
    walkthrough.

    Without an `executor` the evaluations run in-process, on the calling thread,
    and `bundle_size` does not apply. A closed executor raises a
    [`WorkflowError`][ropt.exceptions.WorkflowError] at the first evaluation. A
    run started from inside an evaluation needs an executor with workers of its
    own: the one it is already running on refuses the work. `handlers` takes
    [`EventHandler`][ropt.components.event_handlers.EventHandler] objects, as
    [`optimize`][ropt.simple.optimize] does.

    An evaluation is a single batch that has already run by the time `report`
    sees it, and there is no optimizer loop to interrupt, so unlike on
    [`optimize`][ropt.simple.optimize] returning `True` cannot stop anything: it
    only ends the reporting, and every result is still returned. `metadata` also
    reaches `function` as `context.metadata`.

    Args:
        config:      The optimization configuration.
        variables:   The variable vectors to evaluate, one per row.
        function:    The per-realization evaluation function.
        executor:    The executor to evaluate on, or `None`.
        handlers:    Optional handlers, called in the order listed.
        report:      Optional callback invoked with each evaluation's results.
        bundle_size: Evaluations per worker task, `None` for the executor's own.
        metadata:    Optional dictionary attached to every emitted result.

    Returns:
        One [`FunctionResults`][ropt.results.FunctionResults] per input vector.

    Raises:
        ValueError: If `variables` is not a 2-D matrix.
    """
    array = np.asarray(variables, dtype=np.float64)
    if array.ndim != 2:  # ruff: ignore[magic-value-comparison]
        msg = (
            "evaluate_many() takes a 2-D matrix of vectors (one per row); "
            "use evaluate() for a single vector."
        )
        raise ValueError(msg)
    results = _run_evaluation(
        executor,
        config,
        array,
        function,
        handlers=handlers,
        report=report,
        bundle_size=bundle_size,
        metadata=metadata,
    )
    return tuple(results)


def _run_evaluation(  # ruff: ignore[too-many-arguments]
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
    step = EvaluationStep(evaluator=evaluator)
    step.add_event_handler(history)
    attach_handlers(step, handlers, report)
    step.run(
        context=context,
        variables=np.asarray(variables, dtype=np.float64),
        metadata=metadata,
    )
    return history["results"] or ()
