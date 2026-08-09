"""Run a single optimization with the high-level ``ropt.simple`` API.

This is the simplest optimization: a configuration dictionary, a start vector,
and a per-realization objective callback that returns a scalar. The optimization
directly calls the ``optimize`` function, they do not run in parallel.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ropt.components.evaluators import EvaluationFunctionContext
from ropt.simple import EvaluateResult, optimize

DIM = 5
CONFIG: dict[str, Any] = {
    "variables": {
        "variable_count": DIM,
        "perturbation_magnitudes": 1e-6,
    },
}
INITIAL_VALUES = 2 * np.arange(DIM) / DIM + 0.5


def rosenbrock(
    variables: NDArray[np.float64], _context: EvaluationFunctionContext
) -> float:
    """The multi-dimensional Rosenbrock function, minimized at all ones.

    Args:
        variables: The variable vector to evaluate.

    Returns:
        The Rosenbrock objective at ``variables``.
    """
    objective = 0.0
    for d_idx in range(DIM - 1):
        x, y = variables[d_idx : d_idx + 2]
        objective += (1.0 - x) ** 2 + 100 * (y - x * x) ** 2
    return float(objective)


def report(result: EvaluateResult) -> None:
    """Print each function evaluation as it completes.

    Args:
        result: The result of a single function evaluation.
    """
    if result.target_objective is not None:
        print(f"  objective: {result.target_objective}")


def main() -> None:
    """Run the optimization and check the result."""
    result = optimize(CONFIG, INITIAL_VALUES, rosenbrock, report=report)
    print(f"exit code:         {result.exit_code}")
    print(f"optimal variables: {result.variables}")
    print(f"optimal objective: {result.target_objective}")
    assert result.variables is not None
    assert np.allclose(result.variables, 1.0, atol=1e-2)


if __name__ == "__main__":
    main()
