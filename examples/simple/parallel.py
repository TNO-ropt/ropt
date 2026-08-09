"""Run one optimization with parallel evaluation via ``threads`` or ``processes``.

Opening a ``threads``/``processes`` block fixes a worker pool for the whole
block; the same `optimize` call then evaluates its realizations and gradient
perturbations on that pool. Pass ``-m``/``--multiprocessing`` to use a process
pool instead of threads (the objective must be picklable).
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any

import numpy as np

from ropt.simple import optimize, processes, threads

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ropt.simple import EvaluationFunctionContext

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


def main(*, multiprocessing: bool = False) -> None:
    """Run the optimization on a thread or process pool.

    Args:
        multiprocessing: Use a process pool instead of a thread pool.
    """
    manager = processes if multiprocessing else threads
    with manager(workers=4):
        result = optimize(CONFIG, INITIAL_VALUES, rosenbrock)
    print(f"optimal variables: {result.variables}")
    assert result.variables is not None
    assert np.allclose(result.variables, 1.0, atol=1e-2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--multiprocessing",
        action="store_true",
        help="Use a process pool instead of a thread pool.",
    )
    main(multiprocessing=parser.parse_args().multiprocessing)
