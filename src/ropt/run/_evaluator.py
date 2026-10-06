"""Build the component machinery behind a single run."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ropt.components.evaluators import FunctionEvaluator, ParallelEvaluator

from ._batch_ids import next_batch_id
from ._function import adapt_function

if TYPE_CHECKING:
    from ropt.components.concurrency import AbortSignal
    from ropt.components.evaluators import Evaluator
    from ropt.components.executors import Executor
    from ropt.context import EnOptContext

    from ._function import EvaluationFunction


def make_evaluator(
    context: EnOptContext,
    function: EvaluationFunction,
    executor: Executor | None,
    bundle_size: int | None = None,
    abort_signal: AbortSignal | None = None,
) -> Evaluator:
    n_obj = context.objectives.weights.size
    n_con = (
        0
        if context.nonlinear_constraints is None
        else context.nonlinear_constraints.lower_bounds.size
    )
    callback = adapt_function(function, n_obj, n_con)
    if executor is None:
        return FunctionEvaluator(
            function=callback,
            batch_id_callback=next_batch_id,
            abort_signal=abort_signal,
        )
    return ParallelEvaluator(
        function=callback,
        executor=executor,
        batch_id_callback=next_batch_id,
        bundle_size=bundle_size,
        abort_signal=abort_signal,
    )
