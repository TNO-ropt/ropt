"""Example of optimization of a constrained Rosenbrock function.

This example demonstrates optimization of the a modified multi-dimensional
Rosenbrock function that exhibits uncertainty in its parameters. It shows how to
specify linear and nonlinear constraints. Like the objective, the non-linear
constraint is evaluated by the user-provided function and is stochastic in
nature. The linear constraint is specified in the configuration and is
deterministic.

By default the script only uses a nonlinear constraint. You can add a linear
constraint by passing the `--linear` flag.
"""

import argparse
from functools import partial
from typing import Any

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

from ropt.evaluation import (
    EvaluationBatchContext,
    EvaluationBatchResult,
)
from ropt.results import FunctionResults, Results
from ropt.workflow import BasicOptimizer

DIM = 5
REALIZATIONS = 10
CONFIG: dict[str, Any] = {
    "variables": {
        "variable_count": DIM,
        "perturbation_magnitudes": 1e-6,
    },
    "realizations": {
        "weights": [1.0] * REALIZATIONS,
    },
    "nonlinear_constraints": {
        "lower_bounds": [-np.inf],
        "upper_bounds": [-1.0],
    },
}
INITIAL_VALUES = 2 * np.arange(DIM) / DIM + 0.5
UNCERTAINTY = 0.1


def rosenbrock(
    variables: NDArray[np.float64],
    context: EvaluationBatchContext,
    a: NDArray[np.float64],
    b: NDArray[np.float64],
) -> EvaluationBatchResult:
    """Batch evaluation callback for the multi-dimensional rosenbrock function.

    Args:
        variables: The variables to evaluate.
        context:   Evaluator context.
        a:         The 'a' parameters.
        b:         The 'b' parameters.

    Returns:
        An `EvaluationBatchResult` object containing the calculated objectives.
    """
    objectives = np.zeros((variables.shape[0], 1), dtype=np.float64)
    constraints = np.zeros((variables.shape[0], 1), dtype=np.float64)
    for v_idx, r in enumerate(context.realizations):
        for d_idx in range(DIM - 1):
            x, y = variables[v_idx, d_idx : d_idx + 2]
            objectives[v_idx, 0] += (a[r] - x) ** 2 + b[r] * (y - x * x) ** 2
        constraints[v_idx, 0] += (variables[v_idx, 0] - a[r]) ** 3 - variables[v_idx, 1]
    return EvaluationBatchResult(objectives=objectives, constraints=constraints)


def report(results: tuple[Results, ...]) -> None:
    """Report results of an evaluation.

    Args:
        results: The results to process.
    """
    for item in results:
        if isinstance(item, FunctionResults) and item.functions is not None:
            print(f"  variables:  {item.evaluations.variables}")
            print(f"  objective:  {item.functions.target_objective}")
            print(f"  constraint: {item.functions.constraints}\n")


def main(*, linear: bool = False) -> None:
    """Run the example and check the result."""
    rng = default_rng(seed=123)
    a = rng.normal(loc=1.0, scale=UNCERTAINTY, size=REALIZATIONS)
    b = rng.normal(loc=100.0, scale=100 * UNCERTAINTY, size=REALIZATIONS)

    if linear:
        CONFIG.update(
            {
                "linear_constraints": {
                    "coefficients": [[0.0, 0.0, 0.0, 1.0, -1.0]],
                    "lower_bounds": 0.0,
                    "upper_bounds": 0.0,
                }
            }
        )

    optimizer = BasicOptimizer(
        CONFIG, partial(rosenbrock, a=a, b=b), constraint_tolerance=1e-6
    )
    optimizer.set_results_callback(report)
    optimizer.run(INITIAL_VALUES)
    assert optimizer.results is not None
    assert optimizer.results.functions is not None
    assert optimizer.results.functions.constraints is not None

    print(f"Optimal variables:  {optimizer.results.evaluations.variables}")
    print(f"Optimal objective:  {optimizer.results.functions.target_objective}")
    print(f"Optimal constraint: {optimizer.results.functions.constraints}\n")

    assert np.allclose(optimizer.results.functions.target_objective, 0, atol=1e-1)
    assert np.allclose(optimizer.results.evaluations.variables, 1, atol=1e-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("python rosenbrock_constrained.py")
    parser.add_argument(
        "--linear",
        action="store_true",
        help="add a linear constraint",
    )
    args = parser.parse_args()
    main(linear=args.linear)
