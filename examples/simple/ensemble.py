"""Ensemble optimization with the high-level ``ropt.simple`` API.

An *ensemble* optimization minimizes the mean objective over a set of
realizations with uncertain parameters. Compared to a deterministic run, the
config gains a ``realizations`` section (and, here, a ``gradient`` section), and
the per-realization objective uses ``ctx.realization`` to return the value for
its own realization. A ``report`` callback prints each evaluation as it lands.
"""

from typing import Any

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

from ropt.components.evaluators import EvaluationFunctionContext
from ropt.simple import EvaluateResult, optimize

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
    "gradient": {
        "number_of_perturbations": 5,
    },
}
INITIAL_VALUES = 2 * np.arange(DIM) / DIM + 0.5

_RNG = default_rng(seed=123)
A = _RNG.normal(loc=1.0, scale=UNCERTAINTY, size=REALIZATIONS)
B = _RNG.normal(loc=100.0, scale=100 * UNCERTAINTY, size=REALIZATIONS)


def rosenbrock(
    variables: NDArray[np.float64], context: EvaluationFunctionContext
) -> float:
    """The Rosenbrock function for one realization, minimized at all ones.

    Args:
        variables: The variable vector to evaluate.
        context:   Identifies the realization being evaluated.

    Returns:
        The Rosenbrock objective for ``context.realization``.
    """
    r = context.realization
    objective = 0.0
    for d_idx in range(DIM - 1):
        x, y = variables[d_idx : d_idx + 2]
        objective += (A[r] - x) ** 2 + B[r] * (y - x * x) ** 2
    return float(objective)


def report(result: EvaluateResult) -> None:
    """Print the objective of each function evaluation.

    Args:
        result: The result of a single function evaluation.
    """
    if result.target_objective is not None:
        print(f"  objective: {result.target_objective}")


def main() -> None:
    """Run the ensemble optimization and check the result."""
    result = optimize(CONFIG, INITIAL_VALUES, rosenbrock, report=report)
    print(f"optimal variables: {result.variables}")
    print(f"optimal objective: {result.target_objective}")
    assert result.variables is not None
    assert np.allclose(result.variables, 1.0, atol=1e-1)


if __name__ == "__main__":
    main()
