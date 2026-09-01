"""Ensemble optimization on an HPC cluster with the low-level API.

Advanced counterpart of [hpc.py][]: the same optimization, assembled from
`HPCExecutor`, `ParallelEvaluator` and `OptimizationStep` rather than hidden
behind a session and `optimize`. Every ensemble evaluation is submitted to the
cluster as a job, and the executor's lifetime is managed explicitly, which is
what the high-level API does for you.

Running it needs the `ropt[hpc]` extra, a reachable cluster, and a working
directory on a filesystem the compute nodes share. If you have no cluster
available, pass `--local` to swap `HPCExecutor` for `LocalJobExecutor`, which
runs each evaluation as its own process on this machine: the same job shape
without a cluster, and the local stand-in for the real thing.

Either way the evaluation function goes to an interpreter that cannot import
this script, so this example needs the `ropt[cloudpickle]` extra.

The cluster and queue come from the `pysqa` configuration of the `ropt`
installation unless `--queue` names one; pass `cluster` or `cores` to
`HPCExecutor` to be more explicit still.
"""

import argparse
import asyncio
from pathlib import Path
from typing import Any

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

from ropt.components.compute_steps import OptimizationStep
from ropt.components.evaluators import (
    EvaluationFunctionContext,
    EvaluationFunctionResult,
    ParallelEvaluator,
)
from ropt.components.event_handlers import CallbackHandler, ResultsHandler
from ropt.components.executors import HPCExecutor, LocalJobExecutor
from ropt.context import EnOptContext
from ropt.enums import EnOptEventType
from ropt.events import EnOptEvent
from ropt.results import FunctionResults

DIM = 2
REALIZATIONS = 5
UNCERTAINTY = 0.01
WORKERS = 4
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

_RNG = default_rng(seed=123)
A = _RNG.normal(loc=1.0, scale=UNCERTAINTY, size=REALIZATIONS)
B = _RNG.normal(loc=100.0, scale=100 * UNCERTAINTY, size=REALIZATIONS)


def rosenbrock(
    variables: NDArray[np.float64], context: EvaluationFunctionContext
) -> EvaluationFunctionResult:
    """The Rosenbrock function for one realization, minimized at all ones.

    Args:
        variables: The variable vector to evaluate.
        context:   Identifies the realization being evaluated.

    Returns:
        The Rosenbrock objective for ``context.realization``.
    """
    x, y = variables
    r = context.realization
    return EvaluationFunctionResult(
        objectives=np.asarray([(A[r] - x) ** 2 + B[r] * (y - x * x) ** 2])
    )


def report(event: EnOptEvent) -> None:
    """Print the objective of each function evaluation.

    Args:
        event: The event carrying the results of a finished evaluation.
    """
    for item in event.results:
        if isinstance(item, FunctionResults) and item.functions is not None:
            print(f"  objective: {item.functions.target_objective}", flush=True)


def main(*, workdir: Path, local: bool = False, queue: str | None = None) -> None:
    """Run the optimization with every evaluation running as its own job.

    Args:
        workdir: Existing directory for the job files. On a cluster it must be
                 on a filesystem the compute nodes share.
        local:   Run the jobs on this machine instead of submitting them.
        queue:   The cluster queue to submit to; the configured default if None.
    """
    executor = (
        LocalJobExecutor(workdir=workdir, workers=WORKERS)
        if local
        else HPCExecutor(workdir=workdir, workers=WORKERS, queue=queue)
    )
    step = OptimizationStep(
        evaluator=ParallelEvaluator(
            function=rosenbrock, executor=executor, bundle_size=0 if local else 1
        )
    )
    results = ResultsHandler()
    step.add_event_handler(results)
    step.add_event_handler(
        CallbackHandler(
            callback=report,
            event_types={EnOptEventType.FINISHED_EVALUATION},
        )
    )

    async def _run() -> None:
        async with asyncio.TaskGroup() as tg:
            await executor.start(tg)
            # `run` blocks, so it goes to a thread to leave the loop free to
            # drive the executor that is submitting its evaluations.
            await asyncio.to_thread(
                step.run, EnOptContext.model_validate(CONFIG), INITIAL_VALUES
            )
            executor.cancel()

    asyncio.run(_run())

    optimal_result = results.result
    assert optimal_result is not None
    assert optimal_result.functions is not None
    print(f"optimal variables: {optimal_result.evaluations.variables}", flush=True)
    print(f"optimal objective: {optimal_result.functions.target_objective}", flush=True)
    assert np.allclose(optimal_result.evaluations.variables, 1.0, atol=1e-1)


if __name__ == "__main__":

    def _existing_dir(value: str) -> Path:
        path = Path(value).expanduser().resolve()
        if not path.is_dir():
            msg = f"directory does not exist: {value}"
            raise argparse.ArgumentTypeError(msg)
        return path

    parser = argparse.ArgumentParser("python hpc_executor.py")
    parser.add_argument(
        "workdir",
        type=_existing_dir,
        help="directory for the job files, shared by the compute nodes (must exist)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="run the jobs on this machine instead of submitting them",
    )
    parser.add_argument(
        "--queue",
        default=None,
        help="the cluster queue to submit to (default: the configured queue)",
    )
    main(**vars(parser.parse_args()))
