"""Aggregate results across runs with shared ``handlers()``.

A ``handlers()`` block owns result handlers that aggregate across every run in
the block, sequential or concurrent. Blocks nest, and by default a nested block
*inherits* the enclosing blocks' handlers, so an outer handler also sees the
inner runs; pass ``inherit=False`` to isolate a nested block.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ropt.simple import EvaluationFunctionContext, HistoryHandler, handlers, optimize

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
    """Aggregate across a loop of runs, then show nested inheritance."""
    history = HistoryHandler()
    with handlers(history):
        for start in STARTS:
            optimize(CONFIG, start, rosenbrock)
    print(f"aggregated results across {len(STARTS)} runs: {len(history['results'])}")
    assert history["results"]

    # A nested block inherits the enclosing block's handler by default, so both
    # aggregate the nested run.
    outer = HistoryHandler()
    inner = HistoryHandler()
    with handlers(outer), handlers(inner):
        optimize(CONFIG, INITIAL_VALUES, rosenbrock)
    assert outer["results"]
    assert inner["results"]
    assert len(outer["results"]) == len(inner["results"])


if __name__ == "__main__":
    main()
