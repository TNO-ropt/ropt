"""Restart an optimization from its own best point, collecting every result.

Each call to `optimize` starts a fresh run, so restarting from the previous
best point is just a loop: feed the returned `result.results.variables` back
in as the next start point. The results at that point are passed back as `f0`
and `g0`, so the new run does not evaluate it again. A `report` callback
appending to one list collects every evaluation from every restart, not just
the final one.
"""

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from ropt import EvaluationFunctionContext, optimize

if TYPE_CHECKING:
    from ropt import FunctionResults, GradientResults

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
        The Rosenbrock objective at `variables`.
    """
    objective = 0.0
    for d_idx in range(DIM - 1):
        x, y = variables[d_idx : d_idx + 2]
        objective += (1.0 - x) ** 2 + 100 * (y - x * x) ** 2
    return float(objective)


def main() -> None:
    """Restart from the best point found so far, `RESTARTS` times."""
    reported: list[FunctionResults] = []
    x0 = INITIAL_VALUES
    f0: FunctionResults | None = None
    g0: GradientResults | None = None
    for _ in range(RESTARTS):
        result = optimize(CONFIG, x0, rosenbrock, report=reported.append, f0=f0, g0=g0)
        assert result.results is not None
        x0 = result.results.variables
        f0, g0 = result.results, result.gradient
    print(f"evaluations collected across all restarts: {len(reported)}")
    assert result.results is not None
    best = result.results.target_objective
    print(f"best objective after {RESTARTS} restarts: {best}")
    assert best is not None
    assert best < 1e-4  # ruff: ignore[magic-value-comparison]


if __name__ == "__main__":
    main()
