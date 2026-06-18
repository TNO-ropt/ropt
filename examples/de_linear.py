"""Differential evaluation optimization example.

This example uses the differential evolution method to solve a discrete
problem with a linear constraint.
"""

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
        "lower_bounds": [0.0, 0.0],
        "upper_bounds": [10.0, 10.0],
        "types": VariableType.INTEGER,
    },
    "optimizer": {
        "max_functions": 5,
    },
    "backend": {
        "method": "differential_evolution",
        "options": {"rng": 4},
        "parallel": False,
    },
    "linear_constraints": {
        "coefficients": [1.0, 1.0],
        "lower_bounds": [-np.inf],
        "upper_bounds": [10.0],
    },
}


def function(variables: NDArray[np.float64], _: EvaluatorContext) -> EvaluatorResult:
    """Evaluate the function.

    Args:
        variables: The variables to evaluate.

    Returns:
        An `EvaluatorResult` object containing the calculated objectives.
    """
    x = variables[:, 0]
    y = variables[:, 1]
    objectives = -np.array(np.minimum(3 * x, y), ndmin=2).T
    return EvaluatorResult(objectives=objectives)


def report(results: tuple[Results, ...]) -> None:
    """Report results of an evaluation.

    Args:
        results: The results.
    """
    for item in results:
        if isinstance(item, FunctionResults) and item.functions is not None:
            print(f"  variables: {item.evaluations.variables}")
            print(f"  objective: {item.functions.target_objective}\n")


def main() -> None:
    """Main function."""
    optimizer = BasicOptimizer(CONFIG, function)
    optimizer.set_results_callback(report)
    optimizer.run(INITIAL_VALUES)
    assert optimizer.results is not None
    assert optimizer.results.functions is not None
    assert np.all(optimizer.results.evaluations.variables == [3, 7])
    print(f"Optimal variables: {optimizer.results.evaluations.variables}")
    print(f"Optimal objective: {optimizer.results.functions.target_objective}\n")


if __name__ == "__main__":
    main()
