"""Build the component machinery behind a single run."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ropt._evaluators import make_batch_evaluator

from ._batch_ids import next_batch_id
from ._function import adapt_function

if TYPE_CHECKING:
    from ropt._concurrency import AbortScope
    from ropt._evaluation import EvaluationBatchCallback
    from ropt._executors import Executor
    from ropt.context import EnOptContext

    from ._function import EvaluationFunction


def make_evaluator(
    context: EnOptContext,
    function: EvaluationFunction,
    executor: Executor | None,
    abort_scope: AbortScope | None = None,
) -> EvaluationBatchCallback:
    n_obj = context.objectives.weights.size
    n_con = (
        0
        if context.nonlinear_constraints is None
        else context.nonlinear_constraints.lower_bounds.size
    )
    callback = adapt_function(function, n_obj, n_con)
    return make_batch_evaluator(
        function=callback,
        executor=executor,
        batch_id_callback=next_batch_id,
        abort_scope=abort_scope,
    )
