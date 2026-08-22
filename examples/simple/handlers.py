"""Aggregate results from concurrent runs with shared handler groups.

A local handler belongs to one run at a time, so it cannot safely collect
results from optimizations that run concurrently -- the runs of
``optimize_many``. A group built with ``shared_handlers()`` can: it routes
every run's results through one dispatcher, so several concurrent runs can
feed it safely. A run may feed several groups at once, for different
purposes.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ropt.simple import (
    EvaluationFunctionContext,
    HistoryHandler,
    optimize_many,
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
STARTS = np.array([INITIAL_VALUES, INITIAL_VALUES + 0.1, INITIAL_VALUES - 0.1])


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
    """Collect every result from concurrent runs, in two groups at once."""
    history = HistoryHandler()
    per_run = HistoryHandler()
    with session() as active:
        pool = active.thread_pool(workers=len(STARTS))
        shared = active.shared_handlers(history)
        tagged = active.shared_handlers(per_run)
        optimize_many(CONFIG, STARTS, rosenbrock, pool=pool, handlers=[shared, tagged])
    print(
        f"collected results across {len(STARTS)} concurrent runs: "
        f"{len(history['results'])}"
    )
    assert history["results"]
    assert per_run["results"]
    assert len(history["results"]) == len(per_run["results"])


if __name__ == "__main__":
    main()
