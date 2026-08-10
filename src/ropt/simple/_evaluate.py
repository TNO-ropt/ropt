"""The high-level ``evaluate`` and ``evaluate_many`` entry points."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ropt.components.compute_steps import EvaluationStep
from ropt.components.evaluators import FunctionEvaluator, ParallelEvaluator
from ropt.components.event_handlers import HistoryHandler
from ropt.context import EnOptContext

from ._handlers import current_handlers
from ._objective import adapt_objective
from ._result import EvaluateResult, _build_evaluate_result
from ._session import current_executor, current_session, make_task_namer, run_step

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import ArrayLike

    from ropt.components.executors import Executor
    from ropt.results import FunctionResults

    from ._objective import ObjectiveCallback


def evaluate(
    config: dict[str, Any],
    variables: ArrayLike,
    objective: ObjectiveCallback,
    *,
    metadata: dict[str, Any] | None = None,
) -> EvaluateResult:
    """Evaluate a single variable vector without optimizing.

    Use [`evaluate_many`][ropt.simple.evaluate_many] to evaluate several
    vectors at once. See [Running Optimizations](../running/running.md) for a
    walkthrough.

    Args:
        config:    The optimization configuration.
        variables: The variable vector to evaluate.
        objective: The per-realization objective callback.
        metadata:  An optional dictionary attached to the emitted
                   [`Results`][ropt.results.Results].

    Returns:
        An [`EvaluateResult`][ropt.simple.EvaluateResult] for the vector.

    Raises:
        ValueError: If `variables` is not a single vector.
    """
    array = np.asarray(variables, dtype=np.float64)
    if array.ndim != 1:
        msg = "evaluate() takes a single vector; use evaluate_many() for a batch."
        raise ValueError(msg)
    results = _run_evaluation(current_executor(), config, array, objective, metadata)
    return _build_evaluate_result(results[0])


def evaluate_many(
    config: dict[str, Any],
    variables: ArrayLike,
    objective: ObjectiveCallback,
    *,
    metadata: dict[str, Any] | None = None,
) -> tuple[EvaluateResult, ...]:
    """Evaluate a batch of variable vectors without optimizing.

    Each row of `variables` is one variable vector; the results are returned in
    the same order. See [Running Optimizations](../running/running.md) for a
    walkthrough.

    Args:
        config:    The optimization configuration.
        variables: The variable vectors to evaluate, one per row.
        objective: The per-realization objective callback.
        metadata:  An optional dictionary attached to every emitted
                   [`Results`][ropt.results.Results].

    Returns:
        One [`EvaluateResult`][ropt.simple.EvaluateResult] per input vector.

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
    results = _run_evaluation(current_executor(), config, array, objective, metadata)
    return tuple(_build_evaluate_result(result) for result in results)


def _run_evaluation(
    executor: Executor | None,
    config: dict[str, Any],
    variables: ArrayLike,
    objective: ObjectiveCallback,
    metadata: dict[str, Any] | None = None,
) -> tuple[FunctionResults, ...]:
    context = EnOptContext.model_validate(config)
    n_obj = context.objectives.weights.size
    n_con = (
        0
        if context.nonlinear_constraints is None
        else context.nonlinear_constraints.lower_bounds.size
    )

    callback = adapt_objective(objective, n_obj, n_con)
    get_name = make_task_namer(current_session(), executor)
    evaluator = (
        FunctionEvaluator(function=callback)
        if executor is None
        else ParallelEvaluator(function=callback, executor=executor, get_name=get_name)
    )
    history = HistoryHandler()
    step = EvaluationStep(evaluator=evaluator)
    step.add_event_handler(history)
    shared = current_handlers()
    if shared is not None:
        shared.attach_to(step)

    run_step(
        step,
        context=context,
        variables=np.asarray(variables, dtype=np.float64),
        metadata=metadata,
    )
    return history["results"] or ()
