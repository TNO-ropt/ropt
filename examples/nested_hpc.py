"""Nested Rosenbrock example using an HPC queue for the inner evaluations.

Variant of [nested.py][] that submits each inner evaluation to an HPC queue
(e.g. Slurm) via `HPCServer`. The outer step still runs on a `ThreadingServer`
so multiple inner jobs can be in flight at once.
"""

import argparse
import asyncio
import threading
from functools import partial
from pathlib import Path
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
from ropt.workflow.servers import HPCServer, ThreadingServer

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

THREAD_WORKERS = 2
HPC_WORKERS = 2


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
        objectives=objective, metadata={"worker": f"batch{context.batch_id:04d}"}
    )


def report(event: EnOptEvent) -> None:
    """Print each inner result with its outer thread and inner job names."""
    for item in event.results:
        if isinstance(item, FunctionResults) and item.functions is not None:
            workers = {str(w) for w in item.evaluations.metadata.get("worker", [])}
            thread = item.metadata.get("thread")
            msg = (
                f"batch: {item.batch_id}  thread: {thread}  jobs: {workers}\n"
                f"  variables: {item.evaluations.variables}\n"
                f"  objective: {item.functions.target_objective}\n\n"
            )
            # Single msg with flush to prevent interleaving from threads:
            print(msg, end="", flush=True)


def main(*, hpc_workdir: Path) -> None:
    """Run the nested optimization with inner jobs submitted to HPC.

    Args:
        hpc_workdir: Existing shared-filesystem directory used by
                     `HPCServer` for temporary I/O files.
    """
    rng = default_rng(seed=123)
    a = rng.normal(loc=1.0, scale=UNCERTAINTY, size=REALIZATIONS)
    b = rng.normal(loc=100.0, scale=100 * UNCERTAINTY, size=REALIZATIONS)

    global_results = ResultsHandler()

    # HPC server for the inner evaluations
    inner_server = HPCServer(workdir=hpc_workdir, workers=HPC_WORKERS)
    # Evaluator for the inner optimization, bundling all evaluations in a batch.
    inner_evaluator = AsyncEvaluator(
        function=partial(rosenbrock, a=a, b=b),
        server=inner_server,
        get_name=lambda contexts: f"batch{contexts[0].batch_id:04d}",
        bundle_size=0,
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
        # _optimize call.
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
    # flight at once. Cached so repeated discrete-variable combinations are
    # not re-submitted to the queue.
    outer_server = ThreadingServer(workers=THREAD_WORKERS)
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
    print(f"Optimal batch: {optimal_result.batch_id}", flush=True)
    print(f"Optimal variables: {optimal_result.evaluations.variables}", flush=True)
    print(
        f"Optimal objective: {optimal_result.functions.target_objective}\n", flush=True
    )
    assert np.allclose(optimal_result.functions.target_objective, 0, atol=1e-1)
    assert np.allclose(optimal_result.evaluations.variables, [1, 2, 3, 4], atol=1e-1)


if __name__ == "__main__":

    def _existing_dir_dir(value: str) -> Path:
        path = Path(value).expanduser().resolve()
        if not path.is_dir():
            msg = f"directory does not exist: {value}"
            raise argparse.ArgumentTypeError(msg)
        return path

    parser = argparse.ArgumentParser("python nested_hpc.py")
    parser.add_argument(
        "workdir",
        type=_existing_dir_dir,
        help="shared-filesystem directory for HPCServer I/O (must exist)",
    )
    args = parser.parse_args()
    main(hpc_workdir=args.workdir)
