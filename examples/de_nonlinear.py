"""Differential evaluation optimization example.

This example uses the differential evolution method to solve a discrete
problem with a linear constraint, implemented as a non-linear constraint.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ropt.enums import VariableType
from ropt.evaluator import EvaluatorContext, EvaluatorResult
from ropt.results import FunctionResults, Results
from ropt.workflow import BasicOptimizer

initial_values = 2 * [0.0]
CONFIG: dict[str, Any] = {
    "variables": {
        "variable_count": len(initial_values),
        "types": VariableType.INTEGER,
        "lower_bounds": [0.0, 0.0],
        "upper_bounds": [10.0, 10.0],
    },
    "optimizer": {
        "method": "differential_evolution",
        "options": {"rng": 4},
        "max_functions": 5,
        "parallel": False,
    },
    "nonlinear_constraints": {"lower_bounds": [-np.inf], "upper_bounds": [10.0]},
}


def function(variables: NDArray[np.float64], _: EvaluatorContext) -> EvaluatorResult:
    """Evaluate the function.

    Args:
        variables: The variables to evaluate.

    Returns:
        Calculated objectives and constraints.
    """
    x = variables[:, 0]
    y = variables[:, 1]
    objectives = -np.array(np.minimum(3 * x, y), ndmin=2).T
    constraints = np.array(x + y, ndmin=2).T
    return EvaluatorResult(objectives=objectives, constraints=constraints)


def report(results: tuple[Results, ...]) -> None:
    """Report results of an evaluation.

    Args:
        results: The results.
    """
    for item in results:
        if isinstance(item, FunctionResults) and item.functions is not None:
            print(f"  variables: {item.evaluations.variables}")
            print(f"  objective: {item.functions.weighted_objective}\n")


def run_optimization() -> None:
    """Run the optimization."""
    optimizer = BasicOptimizer(CONFIG, function)
    optimizer.set_results_callback(report)
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert optimizer.results.functions is not None
    assert np.all(optimizer.results.evaluations.variables == [3, 7])
    print(f"Optimal variables: {optimizer.results.evaluations.variables}")
    print(f"Optimal objective: {optimizer.results.functions.weighted_objective}\n")


def main() -> None:
    """Main function."""
    run_optimization()


if __name__ == "__main__":
    main()
