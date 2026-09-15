"""Attach metadata to a run with the high-level ``ropt.simple`` API.

Metadata comes from two independent sources. Passing a ``metadata`` dict to
``optimize`` tags the run: the same dict is copied onto every result as
``result.metadata``. Returning an ``EvaluationFunctionResult`` with a
``metadata`` field instead records per-realization metadata, stored as one array
entry per realization on ``result.evaluations.metadata``.

A per-realization value may also be an array rather than a scalar. It then spans
an extra axis named after the metadata key, which the ``names`` section of the
configuration can label, and which the table handler unstacks into columns.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ropt.simple import (
    DataFrameHandler,
    EvaluationFunctionContext,
    EvaluationFunctionResult,
    optimize,
)

DIM = 3
REALIZATIONS = 3
RUN_ID = 7
SHIFTS = np.array([0.9, 1.0, 1.1])  # one uncertain shift per realization
VARIABLES = ("x", "y", "z")
CONFIG: dict[str, Any] = {
    "variables": {
        "variable_count": DIM,
        "perturbation_magnitudes": 1e-6,
    },
    "realizations": {
        "weights": [1.0] * REALIZATIONS,
    },
    # The `residual` entry labels the axis spanned by the metadata of that name.
    "names": {"variable": VARIABLES, "residual": VARIABLES},
}


def objective(
    variables: NDArray[np.float64], context: EvaluationFunctionContext
) -> EvaluationFunctionResult:
    """Objective for one realization, recording its shift and residual.

    Args:
        variables: The variable vector to evaluate.
        context:   Identifies the realization being evaluated.

    Returns:
        The objective value and the per-realization metadata for this realization.
    """
    shift = SHIFTS[context.realization]
    residual = variables - shift
    return EvaluationFunctionResult(
        objectives=float(np.sum(residual**2)),
        metadata={"shift": shift, "residual": residual},
    )


def main() -> None:
    """Run one optimization, tagging the run and recording per-realization data."""
    table = DataFrameHandler()
    table.add_table(
        "results",
        "functions",
        {
            "batch_id": "Batch",
            "realization": "Realization",
            "evaluations.metadata.shift": "Shift",
            "evaluations.metadata.residual": "Residual",
            "metadata.run_id": "Run ID",
        },
    )
    # `metadata` here is constant, per-run metadata copied onto every result.
    result = optimize(
        CONFIG, np.zeros(DIM), objective, metadata={"run_id": RUN_ID}, handlers=[table]
    )
    print(table["results"])
    best = result.results
    assert best is not None
    print(f"result metadata:          {best.metadata}")
    print(f"per-realization metadata: {best.evaluations.metadata}")

    assert best.metadata["run_id"] == RUN_ID
    assert np.allclose(best.evaluations.metadata["shift"], SHIFTS)
    assert best.evaluations.metadata["residual"].shape == (REALIZATIONS, DIM)


if __name__ == "__main__":
    main()
