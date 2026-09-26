"""Build the low-level machinery behind a single high-level run."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ropt.components.evaluators import FunctionEvaluator, ParallelEvaluator

from ._batch_ids import next_batch_id
from ._function import adapt_function

if TYPE_CHECKING:
    from ropt.components.evaluators import Evaluator
    from ropt.components.executors import Executor
    from ropt.context import EnOptContext

    from ._function import EvaluationFunction


def make_evaluator(
    context: EnOptContext,
    function: EvaluationFunction,
    executor: Executor | None,
    bundle_size: int | None = None,
) -> Evaluator:
    """Wire an evaluator for a validated configuration.

    The number of objectives and constraints the evaluation function must
    produce follows from the context. An executor spreads the evaluations over
    its workers; without one they run in-process on the calling thread, and
    `bundle_size` does not apply.

    Batch IDs come from the program-wide counter either way, so no two runs in
    this process land on the same ID.

    Args:
        context:     The validated optimizer context.
        function:    The user-supplied evaluation function.
        executor:    The executor the evaluations run on, or `None`.
        bundle_size: Evaluations per worker task, `None` for the executor's own.

    Returns:
        The evaluator to run with.
    """
    n_obj = context.objectives.weights.size
    n_con = (
        0
        if context.nonlinear_constraints is None
        else context.nonlinear_constraints.lower_bounds.size
    )
    callback = adapt_function(function, n_obj, n_con)
    if executor is None:
        return FunctionEvaluator(function=callback, batch_id_callback=next_batch_id)
    return ParallelEvaluator(
        function=callback,
        executor=executor,
        batch_id_callback=next_batch_id,
        bundle_size=bundle_size,
    )
