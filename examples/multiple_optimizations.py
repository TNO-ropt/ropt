"""Run multiple optimizations concurrently with `Session.run`.

`run` starts each optimization on a thread of its own, at most `limit` at a
time, and every optimization evaluates on the same pool. Each one is tagged with
a `metadata` dictionary (`label`) that is attached to its results, so the
optimizations can be told apart afterwards.
"""

from functools import partial
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ropt import (
    EvaluationFunctionContext,
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
        The Rosenbrock objective at `variables`.
    """
    objective = 0.0
    for d_idx in range(DIM - 1):
        x, y = variables[d_idx : d_idx + 2]
        objective += (1.0 - x) ** 2 + 100 * (y - x * x) ** 2
    return float(objective)


def main() -> None:
    """Run one optimization per start vector, concurrently, tagging each one."""
    # --8<-- [start:run]
    with session() as s:
        optimize = partial(s.thread_pool(workers=3).optimize, CONFIG)
        results = s.run(
            [
                partial(optimize, start, rosenbrock, metadata={"label": label})
                for label, start in enumerate(STARTS)
            ],
            limit=2,
        )
    # --8<-- [end:run]
    for result in results:
        assert result.results is not None
        label = result.results.metadata["label"]
        objective = result.results.target_objective
        print(f"start {label} -> objective {objective}")
        assert objective is not None
        assert objective < 1.0


if __name__ == "__main__":
    main()
