"""Ensemble optimization with the high-level ``ropt.simple`` API.

An *ensemble* optimization minimizes the mean objective over a set of
realizations with uncertain parameters. Compared to a deterministic run, the
config gains a ``realizations`` section (and, here, a ``gradient`` section), and
the per-realization objective uses ``ctx.realization`` to return the value for
its own realization. A ``report`` callback prints each evaluation as it lands.

Pass ``--merge`` to estimate the gradient from a single perturbation per
realization (``merge_realizations``) instead of several perturbations per
realization.
"""

import argparse
from typing import Any

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

from ropt.components.evaluators import EvaluationFunctionContext
from ropt.simple import EvaluateResult, optimize

DIM = 5
UNCERTAINTY = 0.1
INITIAL_VALUES = 2 * np.arange(DIM) / DIM + 0.5


def report(result: EvaluateResult) -> None:
    """Print the objective of each function evaluation.

    Args:
        result: The result of a single function evaluation.
    """
    if result.target_objective is not None:
        print(f"  objective: {result.target_objective}")


def main(*, merge: bool = False) -> None:
    """Run the ensemble optimization and check the result.

    Args:
        merge: Merge the realizations in the gradient calculation.
    """
    realizations = 50 if merge else 10
    config: dict[str, Any] = {
        "variables": {
            "variable_count": DIM,
            "perturbation_magnitudes": 1e-6,
        },
        "realizations": {
            "weights": [1.0] * realizations,
        },
        "gradient": {
            "number_of_perturbations": 1 if merge else 5,
            "merge_realizations": merge,
        },
    }

    rng = default_rng(seed=123)
    a = rng.normal(loc=1.0, scale=UNCERTAINTY, size=realizations)
    b = rng.normal(loc=100.0, scale=100 * UNCERTAINTY, size=realizations)

    def rosenbrock(
        variables: NDArray[np.float64], context: EvaluationFunctionContext
    ) -> float:
        r = context.realization
        objective = 0.0
        for d_idx in range(DIM - 1):
            x, y = variables[d_idx : d_idx + 2]
            objective += (a[r] - x) ** 2 + b[r] * (y - x * x) ** 2
        return float(objective)

    result = optimize(config, INITIAL_VALUES, rosenbrock, report=report)
    print(f"optimal variables: {result.variables}")
    print(f"optimal objective: {result.target_objective}")
    assert result.variables is not None
    assert np.allclose(result.variables, 1.0, atol=1e-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--merge",
        action="store_true",
        help="merge the realizations in the gradient calculation",
    )
    main(**vars(parser.parse_args()))
