"""Example of optimization of a Rosenbrock test function.

This example demonstrates optimization of the a Rosenbrock function that
exhibits uncertainty in its parameters. Evaluations are submitted to an HPC
queue (e.g. Slurm) via `HPCServer`. For demonstration purposes, the problem and
its parameters have been tuned to make the optimization converge quickly.

Use the `--multiprocessing` flag to run with `MultiprocessingServer` instead,
allowing the example to be tested without access to an HPC cluster.
"""

import argparse
import asyncio
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

from ropt.results import FunctionResults, Results
from ropt.workflow import BasicOptimizer
from ropt.workflow.evaluators import (
    AsyncEvaluator,
    EvaluationFunctionContext,
    EvaluationFunctionResult,
)
from ropt.workflow.servers import HPCServer, MultiprocessingServer, Server

DIM = 2
UNCERTAINTY = 0.01
REALIZATIONS = 5

CONFIG: dict[str, Any] = {
    "variables": {
        "variable_count": DIM,
        "perturbation_magnitudes": 1e-6,
        "lower_bounds": 0.75,
        "upper_bounds": 1.25,
    },
    "realizations": {
        "weights": [1.0] * REALIZATIONS,
    },
    "gradient": {
        "number_of_perturbations": 1,
        "merge_realizations": True,
        "evaluation_policy": "speculative",
    },
    "optimizer": {
        "max_batches": 8,
    },
}
INITIAL_VALUES = [1.1, 1.2]

HPC_WORKERS = 10


def rosenbrock(
    variables: NDArray[np.float64],
    context: EvaluationFunctionContext,
    *,
    a: NDArray[np.float64],
    b: NDArray[np.float64],
) -> EvaluationFunctionResult:
    """Function evaluator for the multi-dimensional rosenbrock function.

    Args:
        variables:    The variables to evaluate.
        context:      The `EvaluationFunctionContext` object identifying the evaluation.
        a:            The 'a' parameters.
        b:            The 'b' parameters.

    Returns:
        The calculated objective.
    """
    x, y = variables
    r = context.realization
    objective = (a[r] - x) ** 2 + b[r] * (y - x * x) ** 2
    return EvaluationFunctionResult(objectives=np.asarray([objective]))


def report(results: tuple[Results, ...]) -> None:
    """Print each inner result with its outer thread and inner job names."""
    for item in results:
        if isinstance(item, FunctionResults) and item.functions is not None:
            print(f"batch: {item.batch_id}")
            print(f"  variables: {item.evaluations.variables}")
            print(f"  objective: {item.functions.target_objective}\n")


def main(*, hpc_workdir: Path | None) -> None:
    """Run the optimization."""
    rng = default_rng(seed=1234)

    # Generate random parameters for the Rosenbrock function
    rng = default_rng(seed=123)
    a = rng.normal(loc=1.0, scale=UNCERTAINTY, size=REALIZATIONS)
    b = rng.normal(loc=100.0, scale=100 * UNCERTAINTY, size=REALIZATIONS)

    # Create the evaluator and server
    server: Server
    if hpc_workdir is not None:
        server = HPCServer(workdir=hpc_workdir, workers=HPC_WORKERS)
    else:
        server = MultiprocessingServer(workers=HPC_WORKERS)
    evaluator = AsyncEvaluator(
        function=partial(rosenbrock, a=a, b=b),
        server=server,
        get_name=lambda context: (
            f"b{context.batch_id}-r{context.realization}-p{context.perturbation}"
        ),
    )

    # Create the basic optimizer
    optimizer = BasicOptimizer(config=CONFIG, evaluator=evaluator)

    # Set the reporter callback
    optimizer.set_results_callback(report)

    # Run the optimization in an asyncio event loop
    async def _run() -> None:
        async with asyncio.TaskGroup() as tg:
            await server.start(tg)
            await asyncio.to_thread(optimizer.run, INITIAL_VALUES)
            server.cancel()

    asyncio.run(_run())

    # Check the results
    assert optimizer.results is not None
    assert optimizer.results.functions is not None
    print(f"Optimal batch: {optimizer.results.batch_id}\n")
    print(f"Optimal variables: {optimizer.results.evaluations.variables}")
    print(f"Optimal objective: {optimizer.results.functions.target_objective}\n")
    assert np.allclose(optimizer.results.functions.target_objective, 0, atol=1e-1)
    assert np.allclose(optimizer.results.evaluations.variables, 1, atol=1e-1)


if __name__ == "__main__":

    def _existing_dir_dir(value: str) -> Path:
        path = Path(value).expanduser().resolve()
        if not path.is_dir():
            msg = f"directory does not exist: {value}"
            raise argparse.ArgumentTypeError(msg)
        return path

    parser = argparse.ArgumentParser("python hpc.py")
    parser.add_argument(
        "workdir",
        nargs="?",
        type=_existing_dir_dir,
        default=None,
        help="shared-filesystem directory for HPCServer I/O (must exist)",
    )
    parser.add_argument(
        "--multiprocessing",
        action="store_true",
        help="use MultiprocessingServer instead of HPCServer",
    )
    args = parser.parse_args()
    if args.multiprocessing:
        main(hpc_workdir=None)
    elif args.workdir is None:
        parser.error("workdir is required unless --multiprocessing is given")
    else:
        main(hpc_workdir=args.workdir)
