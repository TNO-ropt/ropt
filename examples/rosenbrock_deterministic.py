"""Example of optimization of a multi-dimensional Rosenbrock test function.

This example demonstrates optimization of the unmodified (deterministic)
multi-dimensional Rosenbrock function. It shows how to write a minimal
configuration and how to run and monitor the optimization.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ropt.evaluation import EvaluationBatchContext, EvaluationBatchResult
from ropt.results import FunctionResults, Results
from ropt.workflow import BasicOptimizer

DIM = 5
CONFIG: dict[str, Any] = {
    "variables": {
        "variable_count": DIM,
        "perturbation_magnitudes": 1e-6,
    },
}
INITIAL_VALUES = 2 * np.arange(DIM) / DIM + 0.5


def rosenbrock(
    variables: NDArray[np.float64], _: EvaluationBatchContext
) -> EvaluationBatchResult:
    """Function evaluator for the multi-dimensional rosenbrock function.

    Args:
        variables: The variables to evaluate.

    Returns:
        An `EvaluationBatchResult` object containing the calculated objectives.
    """
    objectives = np.zeros((variables.shape[0], 1), dtype=np.float64)
    for v_idx in range(variables.shape[0]):
        for d_idx in range(DIM - 1):
            x, y = variables[v_idx, d_idx : d_idx + 2]
            objectives[v_idx, 0] += (1.0 - x) ** 2 + 100 * (y - x * x) ** 2
    return EvaluationBatchResult(objectives=objectives)


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
    """Run the example and check the result."""
    optimizer = BasicOptimizer(CONFIG, rosenbrock)
    optimizer.set_results_callback(report)
    optimizer.run(INITIAL_VALUES)
    assert optimizer.results is not None
    assert optimizer.results.functions is not None

    print(f"Optimal variables: {optimizer.results.evaluations.variables}")
    print(f"Optimal objective: {optimizer.results.functions.target_objective}\n")

    assert np.allclose(optimizer.results.functions.target_objective, 0, atol=1e-4)
    assert np.allclose(optimizer.results.evaluations.variables, 1, atol=1e-2)


if __name__ == "__main__":
    main()
