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

from ropt.results import FunctionResults, Results
from ropt.workflow import BasicOptimizer
from ropt.workflow.evaluators import FunctionEvaluator

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


def rosenbrock(  # noqa: PLR0913, PLR0917
    variables: NDArray[np.float64],
    realization: int,
    perturbation: int,  # noqa: ARG001
    batch_id: int,  # noqa: ARG001
    eval_idx: int,  # noqa: ARG001
    a: NDArray[np.float64],
    b: NDArray[np.float64],
) -> NDArray[np.float64] | dict[str, Any]:
    """Function evaluator for the multi-dimensional rosenbrock function.

    Args:
        variables:    1-D variable vector for this evaluation.
        realization:  The realization index.
        perturbation: The perturbation index (`-1` when unperturbed).
        batch_id:     Integer identifying the current evaluation batch.
        eval_idx:     Row index within the batch.
        a:         The 'a' parameters.
        b:         The 'b' parameters.

    Returns:
        The evaluation result as an array or a dictionary.
    """
    objective = 0.0
    for d_idx in range(DIM - 1):
        x, y = variables[d_idx : d_idx + 2]
        objective += (a[realization] - x) ** 2 + b[realization] * (y - x * x) ** 2
    return np.array([[objective]], dtype=np.float64)


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

    optimizer = BasicOptimizer(
        CONFIG, FunctionEvaluator(function=partial(rosenbrock, a=a, b=b))
    )
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
