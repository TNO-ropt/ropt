"""Differential evolution optimization example.

This example uses the differential evolution method to solve a discrete
problem with a constraint.

The constraint is linear, but this script shows also how to use a nonlinear
constraint. Use the `--linear` command line argument to switch between linear
and nonlinear constraints:

    usage: python differential_evolution.py [-h] [--linear]

    options:
    -h, --help  show this help message and exit
    --linear    Solve using linear constraints
"""

import argparse
from functools import partial
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ropt.enums import VariableType
from ropt.evaluator import EvaluatorContext, EvaluatorResult
from ropt.results import FunctionResults, Results
from ropt.workflow import BasicOptimizer

INITIAL_VALUES = 2 * [0.0]
CONFIG: dict[str, Any] = {
    "variables": {
        "variable_count": len(INITIAL_VALUES),
        "types": VariableType.INTEGER,
        "lower_bounds": [0.0, 0.0],
        "upper_bounds": [10.0, 10.0],
    },
    "optimizer": {
        "max_functions": 5,
    },
    "backend": {
        "method": "differential_evolution",
        "options": {"rng": 4},
        "parallel": False,
    },
}


def function(
    variables: NDArray[np.float64], _: EvaluatorContext, *, linear: bool = False
) -> EvaluatorResult:
    """Evaluate the function.

    Args:
        variables: The variables to evaluate.
        linear:    Whether to use a linear constraint or a nonlinear constraint.

    Returns:
        An `EvaluatorResult` object containing the calculated objectives and constraints.
    """
    x = variables[:, 0]
    y = variables[:, 1]
    objectives = -np.array(np.minimum(3 * x, y), ndmin=2).T
    constraints = None if linear else np.array(x + y, ndmin=2).T
    return EvaluatorResult(objectives=objectives, constraints=constraints)


def report(results: tuple[Results, ...]) -> None:
    """Report results of an evaluation.

    Args:
        results: The results.
    """
    for item in results:
        if isinstance(item, FunctionResults) and item.functions is not None:
            print(f"  variables: {item.evaluations.variables}")
            print(f"  objective: {item.functions.target_objective}\n")


def main(**kwargs: dict[str, Any]) -> None:
    """Main function."""
    linear = bool(kwargs.get("linear"))
    if linear:
        CONFIG["linear_constraints"] = {
            "coefficients": [1.0, 1.0],
            "lower_bounds": [-np.inf],
            "upper_bounds": [10.0],
        }
    else:
        CONFIG["nonlinear_constraints"] = {
            "lower_bounds": [-np.inf],
            "upper_bounds": [10.0],
        }
    optimizer = BasicOptimizer(CONFIG, partial(function, linear=linear))
    optimizer.set_results_callback(report)
    optimizer.run(INITIAL_VALUES)
    assert optimizer.results is not None
    assert optimizer.results.functions is not None
    assert np.all(optimizer.results.evaluations.variables == [3, 7])
    print(f"Optimal variables: {optimizer.results.evaluations.variables}")
    print(f"Optimal objective: {optimizer.results.functions.target_objective}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("python differential_evolution.py")
    parser.add_argument(
        "--linear",
        action="store_true",
        help="solve using linear constraints",
    )
    main(**vars(parser.parse_args()))
