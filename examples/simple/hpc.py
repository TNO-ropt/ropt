"""Ensemble optimization on an HPC cluster with the high-level ``ropt.simple`` API.

An HPC pool submits evaluations to a cluster queue (through ``pysqa``), so an
``optimize`` call given one runs its ensemble evaluations as cluster jobs. This
is what the example does by default; it needs the ``ropt[hpc]`` extra and a
reachable cluster. The cluster and queue come from the ``pysqa`` configuration
of the ``ropt`` installation unless ``--queue`` names one; other cluster
parameters (such as ``cluster`` and ``cores``) can be passed to ``hpc_pool``
when needed.

If you have no cluster available, pass ``--local`` to run the identical
optimization on a local pool instead. That pool runs each evaluation as its own
process, exactly as a cluster job does, so it is the local stand-in for
``hpc_pool`` and lets the example be exercised anywhere.

Both pools send the evaluation function to a separate interpreter that cannot
import this script, so this example needs the ``ropt[cloudpickle]`` extra.
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

from ropt.simple import (
    EvaluateResult,
    EvaluationFunctionContext,
    optimize,
    session,
)

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
) -> float:
    """The Rosenbrock function for one realization, minimized at all ones.

    Args:
        variables: The variable vector to evaluate.
        context:   Identifies the realization being evaluated.

    Returns:
        The Rosenbrock objective for ``context.realization``.
    """
    x, y = variables
    r = context.realization
    return float((A[r] - x) ** 2 + B[r] * (y - x * x) ** 2)


def report(result: EvaluateResult) -> None:
    """Print the objective of each function evaluation.

    Args:
        result: The result of a single function evaluation.
    """
    if result.target_objective is not None:
        print(f"  objective: {result.target_objective}")


def main(
    *,
    local: bool = False,
    queue: str | None = None,
    workdir: Path | None = None,
) -> None:
    """Run the optimization on the cluster, or on a local pool.

    Args:
        local:   Run on a local pool instead of submitting to a cluster.
        queue:   The cluster queue to submit to; the configured default if None.
        workdir: Directory for the job files. On a cluster it must be on a
                 filesystem the compute nodes share.
    """
    with session() as active:
        pool = (
            active.local_pool(workers=WORKERS, workdir=workdir, bundle_size=0)
            if local
            else active.hpc_pool(workers=WORKERS, queue=queue, workdir=workdir)
        )
        result = optimize(CONFIG, INITIAL_VALUES, rosenbrock, pool=pool, report=report)
    print(f"optimal variables: {result.variables}")
    print(f"optimal objective: {result.target_objective}")
    assert result.variables is not None
    assert np.allclose(result.variables, 1.0, atol=1e-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local",
        action="store_true",
        help="run on a local pool instead of submitting to a cluster",
    )
    parser.add_argument(
        "--queue",
        default=None,
        help="the cluster queue to submit to (default: the configured queue)",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="directory for the job files, shared by the compute nodes",
    )
    main(**vars(parser.parse_args()))
