"""Collect results into tables with `DataFrameHandler`.

The problem is the ensemble Rosenbrock function with variable bounds and a
stochastic nonlinear constraint. Each table is registered with `add_table`,
which takes the table name, the table type, and a dict that maps result fields
to column titles. A `"functions"` table is filled from function evaluations and
a `"gradients"` table from gradient evaluations; one handler holds tables of
both types.
"""

from typing import Any

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

from ropt import DataFrameHandler, EvaluationFunctionContext, optimize

DIM = 2
REALIZATIONS = 5
UNCERTAINTY = 0.1
# --8<-- [start:config]
CONFIG: dict[str, Any] = {
    "variables": {
        "variable_count": DIM,
        "perturbation_magnitudes": 1e-6,
        "lower_bounds": -2.0,
        "upper_bounds": 2.0,
    },
    "realizations": {
        "weights": [1.0] * REALIZATIONS,
    },
    "nonlinear_constraints": {
        "lower_bounds": -np.inf,
        "upper_bounds": 0.0,
    },
    "names": {
        "variable": ("x", "y"),
        "nonlinear_constraint": ("limit",),
    },
}
# --8<-- [end:config]
INITIAL_VALUES = np.array([-1.0, 1.5])

_RNG = default_rng(seed=123)
A = _RNG.normal(loc=1.0, scale=UNCERTAINTY, size=REALIZATIONS)
B = _RNG.normal(loc=100.0, scale=100 * UNCERTAINTY, size=REALIZATIONS)

# --8<-- [start:columns]
FUNCTIONS: dict[str, str] = {
    "batch_id": "Batch",
    "variables": "Variable",
    "target_objective": "Target objective",
    "functions.objectives": "Objective",
    "functions.constraints": "Constraint",
    "constraint_info.nonlinear_violation": "Violation",
}
EVALUATIONS: dict[str, str] = {
    "batch_id": "Batch",
    "realization": "Realization",
    "evaluations.objectives": "Objective",
    "evaluations.constraints": "Constraint",
}
GRADIENTS: dict[str, str] = {
    "batch_id": "Batch",
    "variables": "Variable",
    "gradients.objectives": "Objective gradient",
    "gradients.constraints": "Constraint gradient",
}
PERTURBATIONS: dict[str, str] = {
    "batch_id": "Batch",
    "realization": "Realization",
    "perturbation": "Perturbation",
    "perturbed_variables": "Variable",
    "evaluations.perturbed_objectives": "Objective",
}
# --8<-- [end:columns]


def rosenbrock(
    variables: NDArray[np.float64], context: EvaluationFunctionContext
) -> list[float]:
    """The Rosenbrock objective and nonlinear constraint for one realization.

    Args:
        variables: The variable vector to evaluate.
        context:   Identifies the realization being evaluated.

    Returns:
        The objective followed by the constraint value for the realization.
    """
    r = context.realization
    x, y = variables
    objective = (A[r] - x) ** 2 + B[r] * (y - x * x) ** 2
    constraint = (x - A[r]) ** 3 - y + 1.0
    return [float(objective), float(constraint)]


def main() -> None:
    """Fill two tables of each type, and print them."""
    # --8<-- [start:tables]
    tables = DataFrameHandler()
    tables.add_table("functions", "functions", FUNCTIONS)
    tables.add_table("evaluations", "functions", EVALUATIONS)
    tables.add_table("gradients", "gradients", GRADIENTS)
    tables.add_table("perturbations", "gradients", PERTURBATIONS)
    # --8<-- [end:tables]
    # --8<-- [start:run]
    optimize(CONFIG, INITIAL_VALUES, rosenbrock, handlers=[tables])
    for name, table in tables.get_tables().items():
        print(f"{name}:\n{table}\n")
    # --8<-- [end:run]

    functions = tables["functions"]
    assert functions.columns == [
        "Batch",
        "Variable,x",
        "Variable,y",
        "Target objective",
        "Objective,0",
        "Constraint,limit",
        "Violation,limit",
    ]
    assert tables["evaluations"].height == REALIZATIONS * functions.height
    assert "Constraint gradient,limit,y" in tables["gradients"].columns
    assert tables["perturbations"]["Realization"].n_unique() == REALIZATIONS


if __name__ == "__main__":
    main()
