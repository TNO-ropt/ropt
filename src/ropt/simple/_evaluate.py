"""The high-level ``evaluate`` and ``evaluate_many`` entry points."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ropt.components.compute_steps import EvaluationStep
from ropt.components.event_handlers import HistoryHandler

from ._evaluator import make_evaluator, run_step
from ._handlers import current_handlers
from ._result import EvaluateResult, _build_evaluate_result
from ._session import current_executor

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import ArrayLike

    from ropt.components.executors import Executor
    from ropt.results import FunctionResults

    from ._function import EvaluationFunction


def evaluate(
    config: dict[str, Any],
    variables: ArrayLike,
    function: EvaluationFunction,
    *,
    metadata: dict[str, Any] | None = None,
) -> EvaluateResult:
    """Evaluate a single variable vector without optimizing.

    Use [`evaluate_many`][ropt.simple.evaluate_many] to evaluate several
    vectors at once. See [Running Optimizations](../running/running.md) for a
    walkthrough.

    Unlike [`optimize`][ropt.simple.optimize] this takes no `handlers` or
    `report`: there is no optimization to stop early, and a single batch has
    nothing to accumulate across. An open [`handlers`][ropt.simple.handlers]
    block still receives the results, its `report` callback included.

    Args:
        config:    The optimization configuration.
        variables: The variable vector to evaluate.
        function:  The per-realization evaluation function.
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
    results = _run_evaluation(current_executor(), config, array, function, metadata)
    return _build_evaluate_result(results[0])


def evaluate_many(
    config: dict[str, Any],
    variables: ArrayLike,
    function: EvaluationFunction,
    *,
    metadata: dict[str, Any] | None = None,
) -> tuple[EvaluateResult, ...]:
    """Evaluate a batch of variable vectors without optimizing.

    Each row of `variables` is one variable vector; the results are returned in
    the same order. See [Running Optimizations](../running/running.md) for a
    walkthrough.

    Like [`evaluate`][ropt.simple.evaluate], this takes no `handlers` or
    `report`; an open [`handlers`][ropt.simple.handlers] block still receives
    the results.

    Args:
        config:    The optimization configuration.
        variables: The variable vectors to evaluate, one per row.
        function:  The per-realization evaluation function.
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
    results = _run_evaluation(current_executor(), config, array, function, metadata)
    return tuple(_build_evaluate_result(result) for result in results)


def _run_evaluation(
    executor: Executor | None,
    config: dict[str, Any],
    variables: ArrayLike,
    function: EvaluationFunction,
    metadata: dict[str, Any] | None = None,
) -> tuple[FunctionResults, ...]:
    context, evaluator = make_evaluator(config, function, executor)
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
