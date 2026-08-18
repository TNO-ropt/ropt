"""Evaluate variable vectors without optimizing, via ``evaluate``/``evaluate_many``.

`evaluate` runs a single vector; `evaluate_many` runs the rows of a matrix and
returns one result per row. Neither runs an optimizer — they just compute the
objective (and any constraints) for the vectors you supply.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ropt.simple import EvaluationFunctionContext, evaluate, evaluate_many

DIM = 5
CONFIG: dict[str, Any] = {
    "variables": {
        "variable_count": DIM,
        "perturbation_magnitudes": 1e-6,
    },
}


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
    """Evaluate a single vector and a batch of vectors."""
    single = evaluate(CONFIG, np.ones(DIM), rosenbrock)
    print(f"objective at the optimum: {single.target_objective}")
    assert single.target_objective is not None
    assert np.isclose(single.target_objective, 0.0)

    matrix = np.array([np.zeros(DIM), np.ones(DIM), 2 * np.arange(DIM) / DIM + 0.5])
    batch = evaluate_many(CONFIG, matrix, rosenbrock)
    for vector, result in zip(matrix, batch, strict=True):
        print(f"objective at {vector}: {result.target_objective}")
    assert all(result.target_objective is not None for result in batch)
    optimum = batch[1].target_objective
    assert optimum is not None
    assert np.isclose(optimum, 0.0)


if __name__ == "__main__":
    main()
