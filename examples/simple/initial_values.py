"""Restart with a larger ensemble, reusing the results at the restart point.

A first run optimizes an ensemble of two realizations. A second run restarts
from its best point with five, and is given the function and gradient results
the first run produced there. The two realizations evaluated at that point are
reused, and only the three added ones are evaluated.
"""

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

from ropt.simple import EvaluationFunctionContext, optimize

if TYPE_CHECKING:
    from ropt.results import FunctionResults

DIM = 3
PERTURBATIONS = 4
SCREENING_REALIZATIONS = 2
FULL_REALIZATIONS = 5
UNCERTAINTY = 0.1
INITIAL_VALUES = 2 * np.arange(DIM) / DIM + 0.5

# Drawn once for the full ensemble, so a realization is the same in both runs.
rng = default_rng(seed=123)
a = rng.normal(loc=1.0, scale=UNCERTAINTY, size=FULL_REALIZATIONS)
b = rng.normal(loc=100.0, scale=100 * UNCERTAINTY, size=FULL_REALIZATIONS)


def config(realizations: int) -> dict[str, Any]:
    """Build the configuration for an ensemble of the given size.

    Args:
        realizations: The number of realizations.

    Returns:
        The optimization configuration.
    """
    return {
        "variables": {
            "variable_count": DIM,
            "perturbation_magnitudes": 1e-6,
        },
        "realizations": {"weights": [1.0] * realizations},
        "gradient": {
            "number_of_perturbations": PERTURBATIONS,
            # A gradient at every evaluation, so there is one at the best point.
            "evaluation_policy": "speculative",
        },
        "optimizer": {"max_functions": 5},
    }


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


def main() -> None:
    """Optimize a small ensemble, then restart with a larger one."""
    first = optimize(config(SCREENING_REALIZATIONS), INITIAL_VALUES, rosenbrock)
    assert first.results is not None
    assert first.gradient is not None

    reported: list[FunctionResults] = []
    optimize(
        config(FULL_REALIZATIONS),
        first.results.variables,
        rosenbrock,
        report=reported.append,
        f0=first.results,
        g0=first.gradient,
    )
    restart = reported[0]

    assert first.results.functions is not None
    assert restart.functions is not None
    print(
        f"per-realization objectives at the restart point: "
        f"{restart.evaluations.objectives.ravel()}"
    )
    print(f"reused from the first run: {first.results.evaluations.objectives.ravel()}")
    print(
        f"objective there, {SCREENING_REALIZATIONS} realizations: "
        f"{first.results.functions.objectives}"
    )
    print(
        f"objective there, {FULL_REALIZATIONS} realizations: "
        f"{restart.functions.objectives}"
    )

    assert np.allclose(
        restart.evaluations.objectives[:SCREENING_REALIZATIONS, :],
        first.results.evaluations.objectives,
    )


if __name__ == "__main__":
    main()
