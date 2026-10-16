"""Write result tables to disk while an optimization runs.

A `DataFrameHandler` collects the results into its five default polars tables:
the optimal results, the gradients, the evaluations of each realization, the
evaluations at perturbed points, and the constraint values. Its `write_tables`
method, set as the callback, writes every table to a file each time new results
arrive, so the files show the optimization so far.
"""

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ropt import DataFrameHandler, EvaluationFunctionContext, optimize

OUTPUT_DIR = Path("tables")
TARGETS = np.array([[0.5, 0.8, 1.0], [0.6, 0.9, 1.2], [0.4, 0.7, 1.1]])
# --8<-- [start:config]
CONFIG: dict[str, Any] = {
    "variables": {
        "variable_count": 3,
        "perturbation_magnitudes": 1e-3,
        "lower_bounds": 0.0,
        "upper_bounds": 2.0,
    },
    "realizations": {"weights": [1.0, 1.0, 1.0]},
    "linear_constraints": {
        "coefficients": [[1.0, 1.0, 1.0]],
        "lower_bounds": -np.inf,
        "upper_bounds": 2.4,
    },
    "nonlinear_constraints": {"lower_bounds": -np.inf, "upper_bounds": 0.5},
    "gradient": {"number_of_perturbations": 2},
    "optimizer": {"max_functions": 5, "output_dir": OUTPUT_DIR},
    "names": {
        "variable": ("x", "y", "z"),
        "objective": ("distance",),
        "nonlinear_constraint": ("spread",),
        "linear_constraint": ("total",),
    },
}
# --8<-- [end:config]


def objective(
    variables: NDArray[np.float64], context: EvaluationFunctionContext
) -> list[float]:
    """The distance to the target of a realization, and a constraint value.

    Args:
        variables: The variable vector to evaluate.
        context:   Identifies the realization being evaluated.

    Returns:
        The objective followed by the constraint value for the realization.
    """
    distance = float(np.sum((variables - TARGETS[context.realization]) ** 2))
    spread = float(variables[2] - variables[0])
    return [distance, spread]


# --8<-- [start:handler]
def make_handler() -> DataFrameHandler:
    """Build a handler that keeps the tables and writes them on every update.

    Returns:
        The handler, to pass to `optimize`.
    """
    handler = DataFrameHandler(engine="polars")
    handler.set_default_tables()
    handler.set_callback(handler.write_tables)
    return handler


# --8<-- [end:handler]


def main() -> None:
    """Run the optimization, and print one of the tables it wrote."""
    # --8<-- [start:run]
    optimize(CONFIG, np.ones(3), objective, handlers=[make_handler()])
    # --8<-- [end:run]
    print((OUTPUT_DIR / "functions.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
