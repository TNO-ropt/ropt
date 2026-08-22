"""Restart an optimization from its own best point, collecting every result.

Each call to ``optimize`` starts a fresh run, so restarting from the previous
best point is just a loop: feed the returned ``result.variables`` back in as
the next start point. A ``HistoryHandler`` reused across the loop collects
every result from every restart, not just the final one.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ropt.simple import EvaluationFunctionContext, HistoryHandler, optimize

DIM = 5
CONFIG: dict[str, Any] = {
    "variables": {
        "variable_count": DIM,
        "perturbation_magnitudes": 1e-6,
    },
}
INITIAL_VALUES = 2 * np.arange(DIM) / DIM + 0.5
RESTARTS = 3


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
    """Restart from the best point found so far, `RESTARTS` times."""
    history = HistoryHandler()
    x0 = INITIAL_VALUES
    for _ in range(RESTARTS):
        result = optimize(CONFIG, x0, rosenbrock, handlers=[history])
        assert result.variables is not None
        x0 = result.variables
    print(f"evaluations collected across all restarts: {len(history.results)}")
    print(f"best objective after {RESTARTS} restarts: {result.target_objective}")
    assert result.target_objective is not None
    assert result.target_objective < 1e-4  # ruff: ignore[magic-value-comparison]


if __name__ == "__main__":
    main()
