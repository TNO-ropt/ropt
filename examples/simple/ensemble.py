"""Ensemble optimization with the high-level ``ropt.simple`` API.

An *ensemble* optimization minimizes the mean objective over a set of
realizations with uncertain parameters. Compared to a deterministic run, the
config gains a ``realizations`` section, and the per-realization objective uses
``context.realization`` to return the value for its own realization.
"""

from typing import Any

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

from ropt.simple import EvaluationFunctionContext, optimize

DIM = 5
REALIZATIONS = 10
UNCERTAINTY = 0.1
CONFIG: dict[str, Any] = {
    "variables": {
        "variable_count": DIM,
        "perturbation_magnitudes": 1e-6,
    },
    "realizations": {
        "weights": [1.0] * REALIZATIONS,
    },
}
INITIAL_VALUES = 2 * np.arange(DIM) / DIM + 0.5

rng = default_rng(seed=123)
a = rng.normal(loc=1.0, scale=UNCERTAINTY, size=REALIZATIONS)
b = rng.normal(loc=100.0, scale=100 * UNCERTAINTY, size=REALIZATIONS)


def rosenbrock(
    variables: NDArray[np.float64], context: EvaluationFunctionContext
) -> float:
    """The Rosenbrock function of one realization, with uncertain coefficients.

    Args:
        variables: The variable vector to evaluate.
        context:   Identifies the realization being evaluated.

    Returns:
        The objective of this realization at ``variables``.
    """
    r = context.realization
    objective = 0.0
    for d_idx in range(DIM - 1):
        x, y = variables[d_idx : d_idx + 2]
        objective += (a[r] - x) ** 2 + b[r] * (y - x * x) ** 2
    return float(objective)


def main() -> None:
    """Run the ensemble optimization and check the result."""
    result = optimize(CONFIG, INITIAL_VALUES, rosenbrock)
    print(f"exit code:         {result.exit_code}")
    print(f"optimal variables: {result.variables}")
    print(f"optimal objective: {result.target_objective}")
    assert result.variables is not None
    assert np.allclose(result.variables, 1.0, atol=1e-1)


if __name__ == "__main__":
    main()
