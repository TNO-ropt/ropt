"""Build the low-level machinery behind a single high-level run."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ropt.components.evaluators import FunctionEvaluator, ParallelEvaluator

from ._function import adapt_function

if TYPE_CHECKING:
    from ropt.components.evaluators import Evaluator
    from ropt.context import EnOptContext

    from ._function import EvaluationFunction
    from ._pool import WorkerPool


def make_evaluator(
    context: EnOptContext,
    function: EvaluationFunction,
    pool: WorkerPool,
) -> Evaluator:
    """Wire an evaluator for a validated configuration.

    The number of objectives and constraints the evaluation function must
    produce follows from the context. A pool with an executor spreads the
    evaluations over its workers; a serial pool has none, so they run in-process
    on the calling thread.

    Batch IDs come from the pool's counter either way, so runs sharing a pool
    cannot land on the same ID.

    Args:
        context:  The validated optimizer context.
        function: The user-supplied evaluation function.
        pool:     The pool the evaluations run on.

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
    if pool.executor is None:
        return FunctionEvaluator(function=callback, batch_id_callback=pool.batch_ids)
    return ParallelEvaluator(
        function=callback,
        executor=pool.executor,
        bundle_size=pool.bundle_size,
        batch_id_callback=pool.batch_ids,
    )
