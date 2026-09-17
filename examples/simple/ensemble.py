"""Ensemble optimization with the high-level ``ropt.simple`` API.

An *ensemble* optimization minimizes the mean objective over a set of
realizations with uncertain parameters. Compared to a deterministic run, the
config gains a ``realizations`` section, and the per-realization objective uses
``context.realization`` to return the value for its own realization. A
``report`` callback prints each evaluation as it lands.
"""

from typing import Any

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

from ropt.results import FunctionResults
from ropt.simple import EvaluationFunctionContext, optimize

# --8<-- [start:config]
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
# --8<-- [end:config]

# --8<-- [start:draws]
rng = default_rng(seed=123)
a = rng.normal(loc=1.0, scale=UNCERTAINTY, size=REALIZATIONS)
b = rng.normal(loc=100.0, scale=100 * UNCERTAINTY, size=REALIZATIONS)
# --8<-- [end:draws]


# --8<-- [start:objective]
def rosenbrock(
    variables: NDArray[np.float64], context: EvaluationFunctionContext
) -> float:
    """The Rosenbrock function of one realization, with uncertain coefficients.

    Args:
        variables: The variable vector to evaluate.
        context:   Identifies the realization being evaluated.

    Returns:
        The objective of this realization at `variables`.
    """
    r = context.realization
    objective = 0.0
    for d_idx in range(DIM - 1):
        x, y = variables[d_idx : d_idx + 2]
        objective += (a[r] - x) ** 2 + b[r] * (y - x * x) ** 2
    return float(objective)


# --8<-- [end:objective]


# --8<-- [start:report]
def report(result: FunctionResults) -> None:
    """Print the objective of each evaluation as the run proceeds.

    Args:
        result: The result of a single function evaluation.
    """
    if result.target_objective is not None:
        print(f"  objective: {result.target_objective}")


# --8<-- [end:report]


def main() -> None:
    """Run the ensemble optimization and check the result."""
    # --8<-- [start:run]
    result = optimize(CONFIG, INITIAL_VALUES, rosenbrock, report=report)
    # --8<-- [end:run]
    # --8<-- [start:result]
    print(f"exit code:         {result.exit_code}")
    if result.results is not None:
        print(f"optimal variables: {result.results.variables}")
        print(f"optimal objective: {result.results.target_objective}")
    # --8<-- [end:result]
    assert result.results is not None
    assert np.allclose(result.results.variables, 1.0, atol=1e-1)


if __name__ == "__main__":
    main()
