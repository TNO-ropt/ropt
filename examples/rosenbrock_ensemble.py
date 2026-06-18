"""Example of optimization of a multi-dimensional Rosenbrock test function.

This example demonstrates optimization of the a modified multi-dimensional
Rosenbrock function that exhibits uncertainty in its parameters. It shows how to
write a minimal configuration and how to run and monitor the optimization.
"""

from functools import partial
from typing import Any

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

from ropt.evaluator import EvaluatorContext, EvaluatorResult
from ropt.results import FunctionResults, Results
from ropt.workflow import BasicOptimizer

DIM = 5
UNCERTAINTY = 0.1
REALIZATIONS = 10

CONFIG: dict[str, Any] = {
    "variables": {
        "variable_count": DIM,
        "perturbation_magnitudes": 1e-6,
    },
    "realizations": {
        "weights": [1.0] * REALIZATIONS,
    },
    "gradient": {
        "number_of_perturbations": 5,
    },
}
INITIAL_VALUES = 2 * np.arange(DIM) / DIM + 0.5


def rosenbrock(
    variables: NDArray[np.float64],
    context: EvaluatorContext,
    a: NDArray[np.float64],
    b: NDArray[np.float64],
) -> EvaluatorResult:
    """Function evaluator for the multi-dimensional rosenbrock function.

    Args:
        variables: The variables to evaluate.
        context:   Evaluator context.
        a:         The 'a' parameters.
        b:         The 'b' parameters.

    Returns:
        An `EvaluatorResult` object containing the calculated objectives.
    """
    objectives = np.zeros((variables.shape[0], 1), dtype=np.float64)
    for v_idx, r in enumerate(context.realizations):
        for d_idx in range(DIM - 1):
            x, y = variables[v_idx, d_idx : d_idx + 2]
            objectives[v_idx, 0] += (a[r] - x) ** 2 + b[r] * (y - x * x) ** 2
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
    """Run the example and check the result."""
    rng = default_rng(seed=123)
    a = rng.normal(loc=1.0, scale=UNCERTAINTY, size=REALIZATIONS)
    b = rng.normal(loc=100.0, scale=100 * UNCERTAINTY, size=REALIZATIONS)

    optimizer = BasicOptimizer(CONFIG, partial(rosenbrock, a=a, b=b))
    optimizer.set_results_callback(report)
    optimizer.run(INITIAL_VALUES)
    assert optimizer.results is not None
    assert optimizer.results.functions is not None

    print(f"Optimal variables: {optimizer.results.evaluations.variables}")
    print(f"Optimal objective: {optimizer.results.functions.target_objective}\n")

    assert np.allclose(optimizer.results.functions.target_objective, 0, atol=1e-1)
    assert np.allclose(optimizer.results.evaluations.variables, 1, atol=1e-1)


if __name__ == "__main__":
    main()
