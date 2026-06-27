"""Nested Rosenbrock example using async multiprocessing.

Variant of [nested.py][] that runs the inner evaluations on a
`MultiprocessingServer` (one subprocess per worker) and the outer evaluations on
a `ThreadingServer` (one worker thread per concurrent outer evaluation). Both
are driven through `AsyncEvaluator`.

Each `FunctionResults` is tagged with the OS pid of the worker that computed it
(via `metadata["worker"]`) and the name of the outer worker thread (via
`metadata["thread"]`), so the report can show that inner evaluations are really
running in different subprocesses and that the outer step dispatches them from
different threads.
"""

import asyncio
import os
import threading
from functools import partial
from typing import Any

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

from ropt.context import EnOptContext
from ropt.enums import EnOptEventType, VariableType
from ropt.events import EnOptEvent
from ropt.results import FunctionResults
from ropt.workflow.compute_steps import OptimizationStep
from ropt.workflow.evaluators import (
    AsyncEvaluator,
    CachedEvaluator,
    EvaluationFunctionContext,
    EvaluationFunctionResult,
)
from ropt.workflow.event_handlers import CallbackHandler, HistoryHandler, ResultsHandler
from ropt.workflow.servers import MultiprocessingServer, ThreadingServer

DIM = 4
REALIZATIONS = 10
MASK = [True, True, False, False]
INNER_CONFIG: dict[str, Any] = {
    "variables": {
        "variable_count": DIM,
        "perturbation_magnitudes": 1e-6,
        "mask": MASK,
        "lower_bounds": 0.0,
        "upper_bounds": 10.0,
    },
    "realizations": {"weights": [1.0] * REALIZATIONS},
    "optimizer": {"max_functions": 10},
}

OUTER_CONFIG: dict[str, Any] = {
    "variables": {
        "variable_count": DIM,
        "mask": np.logical_not(MASK),
        "lower_bounds": 0.0,
        "upper_bounds": 10.0,
        "types": VariableType.INTEGER,
    },
    "realizations": {"weights": [1.0]},
    "backend": {
        "method": "differential_evolution",
        "options": {"rng": 4},
        "parallel": True,
        "max_iterations": 10,
    },
}
INITIAL_VALUES = [1.0, 1.0, 1.0, 1.0]
UNCERTAINTY = 0.1


def rosenbrock(
    variables: NDArray[np.float64],
    context: EvaluationFunctionContext,
    a: NDArray[np.float64],
    b: NDArray[np.float64],
) -> EvaluationFunctionResult:
    """Function callback for the multi-dimensional rosenbrock function.

    Args:
        variables:    1-D variable vector for this evaluation.
        context:      The function context.
        a:            The 'a' parameters.
        b:            The 'b' parameters.

    Returns:
        The evaluation result.
    """
    objective = 0.0
    scaled = variables / np.arange(1, DIM + 1)
    for idx in range(DIM - 1):
        x, y = scaled[idx : idx + 2]
        r = context.realization
        objective += (a[r] - x) ** 2 + b[r] * (y - x * x) ** 2
    return EvaluationFunctionResult(
        objectives=objective,
        metadata={"worker": str(os.getpid())},
    )


def report(event: EnOptEvent) -> None:
    """Print each inner result with its outer thread and inner pid."""
    for item in event.results:
        if isinstance(item, FunctionResults) and item.functions is not None:
            workers = {str(w) for w in item.evaluations.metadata.get("worker", [])}
            thread = item.metadata.get("thread")
            print(f"batch: {item.batch_id}  thread: {thread}  pids: {workers}")
            print(f"  variables: {item.evaluations.variables}")
            print(f"  objective: {item.functions.target_objective}\n")


def main() -> None:
    """Run the nested optimization example using async multiprocessing."""
    rng = default_rng(seed=123)
    a = rng.normal(loc=1.0, scale=UNCERTAINTY, size=REALIZATIONS)
    b = rng.normal(loc=100.0, scale=100 * UNCERTAINTY, size=REALIZATIONS)

    global_results = ResultsHandler()

    # Subprocess pool, shared across all outer evaluations.
    inner_server = MultiprocessingServer(workers=2)
    # Evaluator for the inner optimization, bundling all evaluations in a batch.
    inner_evaluator = AsyncEvaluator(
        function=partial(rosenbrock, a=a, b=b), server=inner_server, bundle_size=0
    )

    def _optimize(
        variables: NDArray[np.float64],
        context: EvaluationFunctionContext,  # noqa: ARG001
    ) -> EvaluationFunctionResult:
        new_variables = np.where(MASK, INITIAL_VALUES, variables)

        step = OptimizationStep(evaluator=inner_evaluator)
        result_handler = ResultsHandler()
        step.add_event_handler(result_handler)
        step.add_event_handler(global_results)
        step.add_event_handler(
            CallbackHandler(
                callback=report,
                event_types={EnOptEventType.FINISHED_EVALUATION},
            )
        )

        # Tag every inner result with the outer worker thread that ran this
        # _optimize call, so the report can show parallelism at both layers.
        step.run(
            variables=new_variables,
            context=EnOptContext.model_validate(INNER_CONFIG),
            metadata={"thread": threading.current_thread().name},
        )

        inner_result = result_handler["results"]
        assert inner_result is not None
        assert inner_result.functions is not None
        return EvaluationFunctionResult(
            objectives=np.array(inner_result.functions.target_objective)
        )

    # Outer evaluator: thread pool so multiple inner optimizations are in
    # flight at once. Cached so the discrete-variable combinations seen by the
    # differential evolution optimizer are not re-evaluated.
    outer_server = ThreadingServer(workers=2)
    outer_evaluator = AsyncEvaluator(function=_optimize, server=outer_server)
    history = HistoryHandler()
    cache = CachedEvaluator(
        evaluator=outer_evaluator, hits_key="cached", sources={history}
    )

    outer_step = OptimizationStep(evaluator=cache)
    outer_step.add_event_handler(history)

    outer_context = EnOptContext.model_validate(OUTER_CONFIG)

    async def _run() -> None:
        async with asyncio.TaskGroup() as tg:
            await inner_server.start(tg)
            await outer_server.start(tg)
            await asyncio.to_thread(outer_step.run, outer_context, INITIAL_VALUES)
            outer_server.cancel()
            inner_server.cancel()

    asyncio.run(_run())

    optimal_result = global_results["results"]
    assert optimal_result is not None
    assert optimal_result.functions is not None
    print(f"Optimal batch: {optimal_result.batch_id}")
    print(f"Optimal variables: {optimal_result.evaluations.variables}")
    print(f"Optimal objective: {optimal_result.functions.target_objective}\n")
    assert np.allclose(optimal_result.functions.target_objective, 0, atol=1e-1)
    assert np.allclose(optimal_result.evaluations.variables, [1, 2, 3, 4], atol=1e-1)


if __name__ == "__main__":
    main()
