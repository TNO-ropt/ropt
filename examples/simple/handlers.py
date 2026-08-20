"""Aggregate results across runs with shared handlers.

A group built with ``shared_handlers()`` owns result handlers that aggregate
across every run given the group, sequential or concurrent. A run may feed
several groups at once, and mix them with handlers of its own.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ropt.simple import (
    EvaluationFunctionContext,
    HistoryHandler,
    optimize,
    session,
)

DIM = 5
CONFIG: dict[str, Any] = {
    "variables": {
        "variable_count": DIM,
        "perturbation_magnitudes": 1e-6,
    },
}
INITIAL_VALUES = 2 * np.arange(DIM) / DIM + 0.5
STARTS = np.array([INITIAL_VALUES, INITIAL_VALUES + 0.1])


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
    """Aggregate across a loop of runs, then feed two groups at once."""
    history = HistoryHandler()
    with session() as active:
        shared = active.shared_handlers(history)
        for start in STARTS:
            optimize(CONFIG, start, rosenbrock, handlers=[shared])
    print(f"aggregated results across {len(STARTS)} runs: {len(history['results'])}")
    assert history["results"]

    # A run feeds every group it is given, so several groups can collect the
    # same run for different purposes.
    per_project = HistoryHandler()
    per_case = HistoryHandler()
    with session() as active:
        project = active.shared_handlers(per_project)
        case = active.shared_handlers(per_case)
        optimize(CONFIG, INITIAL_VALUES, rosenbrock, handlers=[project, case])
    assert per_project["results"]
    assert per_case["results"]
    assert len(per_project["results"]) == len(per_case["results"])


if __name__ == "__main__":
    main()
