"""Run an optimization on an HPC cluster with ``hpc``.

Opening an ``hpc`` block fixes an HPC worker pool for the block, so the same
`optimize` call submits its evaluations as cluster jobs. This requires the
``ropt[hpc]`` extra and a reachable cluster. The script uses the cluster that is
defined as the default in the installation.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ropt.components.evaluators import EvaluationFunctionContext
from ropt.simple import hpc, optimize

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


def main() -> None:
    """Run the optimization on the cluster."""
    with hpc(workers=10):
        result = optimize(CONFIG, INITIAL_VALUES, rosenbrock)
    print(f"optimal variables: {result.variables}")
    print(f"optimal objective: {result.target_objective}")


if __name__ == "__main__":
    main()
