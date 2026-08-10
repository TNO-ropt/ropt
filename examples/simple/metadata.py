"""Attach metadata to a run with the high-level ``ropt.simple`` API.

Metadata comes from two independent sources. Passing a ``metadata`` dict to
``optimize`` tags the run: the same dict is copied onto every result as
``result.metadata``. Returning an ``EvaluationFunctionResult`` with a
``metadata`` field instead records per-realization metadata, stored as one array
entry per realization on ``result.evaluations.metadata``.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ropt.simple import (
    EvaluationFunctionContext,
    EvaluationFunctionResult,
    optimize,
)

DIM = 3
REALIZATIONS = 3
RUN_ID = 7
SHIFTS = np.array([0.9, 1.0, 1.1])  # one uncertain shift per realization
CONFIG: dict[str, Any] = {
    "variables": {
        "variable_count": DIM,
        "perturbation_magnitudes": 1e-6,
    },
    "realizations": {
        "weights": [1.0] * REALIZATIONS,
    },
}


def objective(
    variables: NDArray[np.float64], context: EvaluationFunctionContext
) -> EvaluationFunctionResult:
    """Objective for one realization, recording its shift as metadata.

    Args:
        variables: The variable vector to evaluate.
        context:   Identifies the realization being evaluated.

    Returns:
        The objective value and the per-realization metadata for this realization.
    """
    shift = SHIFTS[context.realization]
    value = float(np.sum((variables - shift) ** 2))
    return EvaluationFunctionResult(objectives=value, metadata={"shift": shift})


def main() -> None:
    """Run one optimization, tagging the run and recording per-realization data."""
    # `metadata` here is constant, per-run metadata copied onto every result.
    result = optimize(CONFIG, np.zeros(DIM), objective, metadata={"run_id": RUN_ID})

    best = result.results
    assert best is not None
    print(f"result metadata:          {best.metadata}")
    print(f"per-realization metadata: {best.evaluations.metadata}")

    assert best.metadata["run_id"] == RUN_ID
    assert np.allclose(best.evaluations.metadata["shift"], SHIFTS)


if __name__ == "__main__":
    main()
