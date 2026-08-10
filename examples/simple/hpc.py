"""Ensemble optimization on an HPC cluster with the high-level ``ropt.simple`` API.

Opening an ``hpc`` block fixes an HPC worker pool (through ``pysqa``) for the
block, so the same ``optimize`` call submits its ensemble evaluations as cluster
jobs. It uses the default cluster and queue from the ``pysqa`` configuration of
the ``ropt`` installation; cluster-specific parameters (such as ``cluster``,
``queue``, and ``cores``) can be passed to ``hpc`` when needed. Running it needs
the ``ropt[hpc]`` extra and a reachable cluster; pass ``--multiprocessing`` to
run the identical optimization on a local process pool instead, which needs no
cluster and lets the example be exercised anywhere.
"""

import argparse
from typing import Any

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

from ropt.simple import (
    EvaluateResult,
    EvaluationFunctionContext,
    hpc,
    optimize,
    processes,
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


def main(*, multiprocessing: bool = False) -> None:
    """Run the optimization on the cluster (or a local process pool).

    Args:
        multiprocessing: Run on a local process pool instead of an HPC cluster.
    """
    manager = processes(workers=WORKERS) if multiprocessing else hpc(workers=WORKERS)
    with manager:
        result = optimize(CONFIG, INITIAL_VALUES, rosenbrock, report=report)
    print(f"optimal variables: {result.variables}")
    print(f"optimal objective: {result.target_objective}")
    assert result.variables is not None
    assert np.allclose(result.variables, 1.0, atol=1e-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--multiprocessing",
        action="store_true",
        help="run on a local process pool instead of an HPC cluster",
    )
    main(**vars(parser.parse_args()))
