"""Constrained integer optimization with differential evolution.

A small integer problem solved with the gradient-free *differential evolution*
backend: maximize ``min(3 * x, y)`` (by minimizing its negation) over two
integer variables subject to ``x + y <= 10``. Integer variables are declared
through ``variables.types``. Pass ``--linear`` to impose the ``x + y <= 10``
bound as a deterministic *linear* constraint (declared in the config) instead of
a *nonlinear* one (returned by the objective).
"""

import argparse
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ropt.components.evaluators import EvaluationFunctionContext
from ropt.enums import VariableType
from ropt.simple import EvaluateResult, optimize

INITIAL_VALUES = [0.0, 0.0]


def report(result: EvaluateResult) -> None:
    """Print the variables and objective of each function evaluation.

    Args:
        result: The result of a single function evaluation.
    """
    if result.target_objective is not None:
        print(f"  variables: {result.results.evaluations.variables}")
        print(f"  objective: {result.target_objective}")


def main(*, linear: bool = False) -> None:
    """Run the differential evolution optimization and check the result.

    Args:
        linear: Impose ``x + y <= 10`` as a linear rather than nonlinear
                constraint.
    """
    config: dict[str, Any] = {
        "variables": {
            "variable_count": 2,
            "types": VariableType.INTEGER,
            "lower_bounds": [0.0, 0.0],
            "upper_bounds": [10.0, 10.0],
        },
        "optimizer": {
            "max_functions": 5,
        },
        "backend": {
            "method": "differential_evolution",
            "options": {"rng": 4},
            "parallel": False,
        },
    }
    if linear:
        config["linear_constraints"] = {
            "coefficients": [1.0, 1.0],
            "lower_bounds": [-np.inf],
            "upper_bounds": [10.0],
        }
    else:
        config["nonlinear_constraints"] = {
            "lower_bounds": [-np.inf],
            "upper_bounds": [10.0],
        }

    def function(
        variables: NDArray[np.float64],
        _context: EvaluationFunctionContext,
    ) -> float | list[float]:
        x, y = variables
        objective = -min(3.0 * x, y)
        if linear:
            return float(objective)
        return [float(objective), float(x + y)]

    result = optimize(config, INITIAL_VALUES, function, report=report)
    print(f"optimal variables: {result.variables}")
    print(f"optimal objective: {result.target_objective}")
    assert result.variables is not None
    assert np.all(result.variables == [3, 7])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--linear",
        action="store_true",
        help="solve using a linear constraint",
    )
    main(**vars(parser.parse_args()))
