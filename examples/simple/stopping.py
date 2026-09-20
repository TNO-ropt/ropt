"""Stopping an optimization from the ``report`` callback.

A run normally ends when the optimizer converges or hits its own budget. The
``report`` callback sees every result as it arrives, so it can also end the run
itself: returning ``True`` stops the run, and it finishes with
``USER_ABORT``.

Stopping this way is graceful rather than abrupt. The run keeps the best result
it has found, so a stopped run still returns a usable answer -- unlike a run
that fails, which returns nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ropt.enums import ExitCode
from ropt.simple import optimize

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ropt.results import FunctionResults
    from ropt.simple import EvaluationFunctionContext

DIM = 5
MAX_RESULTS = 4  # stop once this many results have arrived
CONFIG: dict[str, Any] = {
    "variables": {
        "variable_count": DIM,
        "perturbation_magnitudes": 1e-6,
    },
    # Enough evaluations that the callback, not the optimizer, ends the run.
    "optimizer": {"max_functions": 50},
}
INITIAL_VALUES = np.zeros(DIM)

_seen = 0  # results the callback has been given so far


def objective(
    variables: NDArray[np.float64], _context: EvaluationFunctionContext
) -> float:
    """The Rosenbrock function, which takes many evaluations to minimize.

    Args:
        variables: The variable vector to evaluate.

    Returns:
        The objective at ``variables``.
    """
    total = 0.0
    for idx in range(DIM - 1):
        x, y = variables[idx : idx + 2]
        total += (1.0 - x) ** 2 + 100 * (y - x * x) ** 2
    return float(total)


def stop_after_max_results(result: FunctionResults) -> bool:
    """Count one result and decide whether the run should go on.

    Args:
        result: The result that just arrived.

    Returns:
        True once the maximum is reached, which stops the run.
    """
    global _seen  # ruff: ignore[global-statement]
    _seen += 1
    print(f"result {_seen}: objective={result.target_objective}")
    return _seen >= MAX_RESULTS


def main() -> None:
    """Run until the callback calls a halt, then read the best point found."""
    result = optimize(CONFIG, INITIAL_VALUES, objective, report=stop_after_max_results)

    print(f"exit_code={result.exit_code.name} after {_seen} results")
    assert result.exit_code == ExitCode.USER_ABORT
    assert _seen == MAX_RESULTS

    # Stopping keeps what the run had already found.
    assert result.results is not None
    assert result.results.target_objective is not None
    print(f"best objective so far: {result.results.target_objective}")


if __name__ == "__main__":
    main()
