"""Differential evaluation optimization example.

This example uses the differential evolution method to solve a discrete
problem with a linear constraint, implemented as a non-linear constraint.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ropt.enums import ConstraintType, EventType, VariableType
from ropt.evaluator import EvaluatorContext, EvaluatorResult
from ropt.plan import BasicOptimizer, Event
from ropt.results import FunctionResults

CONFIG: dict[str, Any] = {
    "variables": {
        "initial_values": 2 * [0.0],
        "types": VariableType.INTEGER,
        "lower_bounds": [0.0, 0.0],
        "upper_bounds": [10.0, 10.0],
    },
    "optimizer": {
        "method": "differential_evolution",
        "options": {"seed": 123},
        "max_functions": 10,
        "parallel": True,
    },
    "nonlinear_constraints": {
        "types": [ConstraintType.LE],
        "rhs_values": [10.0],
    },
}


def function(variables: NDArray[np.float64], _: EvaluatorContext) -> EvaluatorResult:
    """Evaluate the function.

    Args:
        variables: The variables to evaluate.
        context:   Evaluator context.

    Returns:
        Calculated objectives and constraints.
    """
    x = variables[:, 0]
    y = variables[:, 1]
    objectives = -np.array(np.minimum(3 * x, y), ndmin=2).T
    constraints = np.array(x + y, ndmin=2).T
    return EvaluatorResult(objectives=objectives, constraints=constraints)


def report(event: Event) -> None:
    """Report results of an evaluation.

    Args:
        event: event data.
    """
    for item in event.data["results"]:
        if isinstance(item, FunctionResults) and item.functions is not None:
            print(f"  variables: {item.evaluations.variables}")
            print(f"  objective: {item.functions.weighted_objective}\n")


def run_optimization() -> None:
    """Run the optimization."""
    optimal_result = (
        BasicOptimizer(CONFIG, function)
        .add_observer(EventType.FINISHED_EVALUATION, report)
        .run()
        .results
    )
    assert optimal_result is not None
    assert optimal_result.functions is not None
    assert np.all(optimal_result.evaluations.variables == [3, 7])
    print(f"  variables: {optimal_result.evaluations.variables}")
    print(f"  objective: {optimal_result.functions.weighted_objective}\n")


def main() -> None:
    """Main function."""
    run_optimization()


if __name__ == "__main__":
    main()
