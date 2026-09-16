"""What a failing realization does to a run, and how to allow some failures.

An evaluation reports failure by returning ``NaN`` for a realization; nothing
raises. The optimizer needs a minimum number of successful realizations to form
an aggregate, set by ``realization_min_success``, which defaults to *all* of
them. So one ``NaN`` is already enough to end a run with
``TOO_FEW_REALIZATIONS`` and no result at all.

This example runs the same problem twice: once with the default, where a single
failing realization stops everything, and once allowing that failure, where the
run finishes on the realizations that did work.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ropt.enums import ExitCode
from ropt.simple import optimize

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ropt.simple import EvaluationFunctionContext

DIM = 3
REALIZATIONS = 4
FAILING = 2  # the one realization that cannot be evaluated

CONFIG: dict[str, Any] = {
    "variables": {
        "variable_count": DIM,
        "perturbation_magnitudes": 1e-6,
    },
    "realizations": {"weights": [1.0] * REALIZATIONS},
    "optimizer": {"max_functions": 5},
}
INITIAL_VALUES = np.zeros(DIM)


def objective(
    variables: NDArray[np.float64], context: EvaluationFunctionContext
) -> float:
    """A quadratic that one realization cannot evaluate.

    Args:
        variables: The variable vector to evaluate.
        context:   Identifies the realization being evaluated.

    Returns:
        The objective, or ``NaN`` for the realization that fails.
    """
    if context.realization == FAILING:
        return float("nan")
    return float(np.sum((variables - 1.0) ** 2))


def main() -> None:
    """Run with the default minimum, then with the failure allowed."""
    # Every realization must succeed, so the single NaN ends the run and
    # leaves every field of the result unset.
    strict = optimize(CONFIG, INITIAL_VALUES, objective)
    print(f"all required: {strict.exit_code.name}, variables={strict.variables}")
    assert strict.exit_code == ExitCode.TOO_FEW_REALIZATIONS
    assert strict.variables is None
    assert strict.target_objective is None

    # Allowing one failure lets the aggregate form from the rest.
    CONFIG["realizations"]["realization_min_success"] = REALIZATIONS - 1
    lenient = optimize(CONFIG, INITIAL_VALUES, objective)
    print(f"one allowed: {lenient.exit_code.name}, variables={lenient.variables}")
    assert lenient.exit_code != ExitCode.TOO_FEW_REALIZATIONS
    assert lenient.variables is not None
    assert np.allclose(lenient.variables, 1.0, atol=1e-1)


if __name__ == "__main__":
    main()
