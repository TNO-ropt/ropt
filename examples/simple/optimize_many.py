"""Run several optimizations concurrently with ``optimize_many``.

`optimize_many` runs a batch of optimizations on driver threads that all
evaluate on the pool it is given. Any of ``config``/``x0``/``objective`` may be a
single value (broadcast to every run) or a per-run sequence; here a matrix of
start vectors sets the number of runs while the config and objective are re-used
by all runs. Each run is tagged with a ``metadata`` dictionary (``run_id``) that
is attached to its results, so the runs can be told apart afterwards.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ropt.simple import EvaluationFunctionContext, optimize_many, session

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
    """Run one optimization per start vector, concurrently, tagging each run."""
    run_metadata = [{"run_id": idx} for idx in range(len(STARTS))]
    with session() as active:
        results = optimize_many(
            CONFIG,
            STARTS,
            rosenbrock,
            pool=active.thread_pool(workers=3),
            metadata=run_metadata,
            limit=2,
        )
    for result in results:
        assert result.results is not None
        run_id = result.results.metadata["run_id"]
        print(f"run {run_id} -> objective {result.target_objective}")
        assert result.target_objective is not None
        assert result.target_objective < 1.0


if __name__ == "__main__":
    main()
