"""Mixed-integer optimization with the high-level ``ropt.simple`` API.

Two of the four variables are continuous and two are discrete (integer-valued),
so the problem is solved with a gradient-free *differential evolution* backend
selected in the config. Discreteness is declared through ``variables.types``;
everything else is the same ensemble Rosenbrock setup. A ``report`` callback
prints each evaluation as the search proceeds.
"""

from typing import Any

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

from ropt.components.evaluators import EvaluationFunctionContext
from ropt.enums import VariableType
from ropt.simple import EvaluateResult, optimize

DIM = 4
REALIZATIONS = 10
UNCERTAINTY = 0.1
CONFIG: dict[str, Any] = {
    "variables": {
        "variable_count": DIM,
        "perturbation_magnitudes": 1e-6,
        "lower_bounds": 0.0,
        "upper_bounds": 10.0,
        "types": [
            VariableType.REAL,
            VariableType.REAL,
            VariableType.INTEGER,
            VariableType.INTEGER,
        ],
    },
    "backend": {
        "method": "differential_evolution",
        "options": {"rng": 4},
        "max_iterations": 50,
    },
    "realizations": {
        "weights": [1.0] * REALIZATIONS,
    },
}
INITIAL_VALUES = [1.0, 1.0, 1.0, 1.0]

_RNG = default_rng(seed=123)
A = _RNG.normal(loc=1.0, scale=UNCERTAINTY, size=REALIZATIONS)
B = _RNG.normal(loc=100.0, scale=100 * UNCERTAINTY, size=REALIZATIONS)


def rosenbrock(
    variables: NDArray[np.float64], context: EvaluationFunctionContext
) -> float:
    """The Rosenbrock function for one realization, minimized at ``[1, 2, 3, 4]``.

    Args:
        variables: The variable vector to evaluate.
        context:   Identifies the realization being evaluated.

    Returns:
        The Rosenbrock objective for ``context.realization``.
    """
    r = context.realization
    objective = 0.0
    scaled = variables / np.arange(1, DIM + 1)
    for idx in range(DIM - 1):
        x, y = scaled[idx : idx + 2]
        objective += (A[r] - x) ** 2 + B[r] * (y - x * x) ** 2
    return float(objective)


def report(result: EvaluateResult) -> None:
    """Print the variables and objective of each function evaluation.

    Args:
        result: The result of a single function evaluation.
    """
    if result.target_objective is not None:
        print(f"  variables: {result.results.evaluations.variables}")
        print(f"  objective: {result.target_objective}")


def main() -> None:
    """Run the mixed-integer optimization and check the result."""
    result = optimize(CONFIG, INITIAL_VALUES, rosenbrock, report=report)
    print(f"optimal variables: {result.variables}")
    print(f"optimal objective: {result.target_objective}")
    assert result.variables is not None
    assert np.allclose(result.variables, [1, 2, 3, 4], atol=1e-1)


if __name__ == "__main__":
    main()
