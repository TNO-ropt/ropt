"""Constrained ensemble optimization with the high-level ``ropt.simple`` API.

The problem adds a stochastic *nonlinear constraint* to the ensemble Rosenbrock
function. In the high-level API a single objective callback returns the
objective **and** the constraint (objectives first, then constraints); the
config declares the constraint bounds under ``nonlinear_constraints``, and
``constraint_tolerance`` sets when a constraint counts as satisfied. A
``report`` callback flags any evaluation that violates the constraint. Pass
``--linear`` to additionally impose a deterministic linear equality constraint
(declared in the config rather than returned by the objective).
"""

import argparse
from typing import Any

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

from ropt.simple import EvaluateResult, EvaluationFunctionContext, optimize

DIM = 5
REALIZATIONS = 10
UNCERTAINTY = 0.1
CONFIG: dict[str, Any] = {
    "variables": {
        "variable_count": DIM,
        "perturbation_magnitudes": 1e-6,
        "lower_bounds": -5.0,
        "upper_bounds": 5.0,
    },
    "realizations": {
        "weights": [1.0] * REALIZATIONS,
    },
    "nonlinear_constraints": {
        "lower_bounds": -np.inf,
        "upper_bounds": -1.0,
    },
}
INITIAL_VALUES = 2 * np.arange(DIM) / DIM + 0.5

_RNG = default_rng(seed=123)
A = _RNG.normal(loc=1.0, scale=UNCERTAINTY, size=REALIZATIONS)
B = _RNG.normal(loc=100.0, scale=100 * UNCERTAINTY, size=REALIZATIONS)


def rosenbrock(
    variables: NDArray[np.float64], context: EvaluationFunctionContext
) -> list[float]:
    """The Rosenbrock objective and nonlinear constraint for one realization.

    Args:
        variables: The variable vector to evaluate.
        context:   Identifies the realization being evaluated.

    Returns:
        The objective followed by the constraint value for the realization.
    """
    r = context.realization
    objective = 0.0
    for d_idx in range(DIM - 1):
        x, y = variables[d_idx : d_idx + 2]
        objective += (A[r] - x) ** 2 + B[r] * (y - x * x) ** 2
    x, y = variables[:2]
    constraint = (x - A[r]) ** 3 - y
    return [float(objective), float(constraint)]


def report(result: EvaluateResult) -> None:
    """Print any constraint violation, and the point that caused it.

    Args:
        result: The result of a single function evaluation.
    """
    info = result.results.constraint_info
    if (
        info is not None
        and info.nonlinear_violation is not None
        and np.any(info.nonlinear_violation > 0)
    ):
        print(f"  constraint violation: {info.nonlinear_violation}")
        print(f"  at variables: {result.variables}")


def main(*, linear: bool = False) -> None:
    """Run the constrained optimization and check the result.

    Args:
        linear: Also add a deterministic linear equality constraint.
    """
    config = {**CONFIG}
    if linear:
        config["linear_constraints"] = {
            "coefficients": [[0.0, 0.0, 0.0, 1.0, -1.0]],
            "lower_bounds": 0.0,
            "upper_bounds": 0.0,
        }
    result = optimize(
        config, INITIAL_VALUES, rosenbrock, report=report, constraint_tolerance=1e-6
    )
    print(f"optimal variables:  {result.variables}")
    print(f"optimal objective:  {result.target_objective}")
    print(f"optimal constraint: {result.constraints}")
    assert result.variables is not None
    assert np.allclose(result.variables, 1.0, atol=1e-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--linear",
        action="store_true",
        help="add a linear equality constraint",
    )
    main(**vars(parser.parse_args()))
