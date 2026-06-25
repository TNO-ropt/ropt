"""Example of optimization of a multi-dimensional Rosenbrock test function.

This example demonstrates optimization of the a modified multi-dimensional
Rosenbrock function that exhibits uncertainty in its parameters. It shows how to
write a minimal configuration and how to run and monitor the optimization.

This script demonstrates the use of the `FunctionEvaluator` to optimize the
function using a function callback.

This script demonstrate how to use either multiple perturbations or only a
single one with the `merge_realizations` option. You can select between the two
options using a command line argument:

    usage: python rosenbrock.py [-h] [--merge] [--function] [--workflow]

    options:
    -h, --help  show this help message and exit
    --merge     Merge the realizations in gradient calculation
"""

import argparse
from functools import partial
from typing import Any

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

from ropt.results import FunctionResults, Results
from ropt.workflow import BasicOptimizer
from ropt.workflow.evaluators import (
    EvaluatorFunctionContext,
    EvaluatorFunctionResult,
    FunctionEvaluator,
)

DIM = 5
CONFIG: dict[str, Any] = {
    "variables": {
        "variable_count": DIM,
        "perturbation_magnitudes": 1e-6,
    },
}
INITIAL_VALUES = 2 * np.arange(DIM) / DIM + 0.5
UNCERTAINTY = 0.1


def rosenbrock(
    variables: NDArray[np.float64],
    context: EvaluatorFunctionContext,
    *,
    a: NDArray[np.float64],
    b: NDArray[np.float64],
) -> EvaluatorFunctionResult:
    """Function callback for the multi-dimensional rosenbrock function.

    Args:
        variables:    1-D variable vector for this evaluation.
        context:      The function context.
        a:            The 'a' parameters.
        b:            The 'b' parameters.

    Returns:
        The evaluation result.
    """
    objective = 0.0
    for idx in range(DIM - 1):
        x, y = variables[idx : idx + 2]
        objective += (a[context.realization] - x) ** 2 + b[context.realization] * (
            y - x * x
        ) ** 2
    return EvaluatorFunctionResult(objective)


def report(results: tuple[Results, ...]) -> None:
    """Report results of an evaluation.

    Args:
        results: The results to process.
    """
    for item in results:
        if isinstance(item, FunctionResults) and item.functions is not None:
            print(f"  variables: {item.evaluations.variables}")
            print(f"  objective: {item.functions.target_objective}\n")


def main(*, merge: bool = False) -> None:
    """Run the example and check the result."""
    # Set the number of realizations and the merge option
    realizations = 50 if merge else 10
    CONFIG.update(
        {
            "realizations": {
                "weights": [1.0] * realizations,
            },
            "gradient": {
                "number_of_perturbations": 1 if merge else 5,
                "merge_realizations": merge,
            },
        }
    )

    # Generate random parameters for the Rosenbrock function
    rng = default_rng(seed=123)
    a = rng.normal(loc=1.0, scale=UNCERTAINTY, size=realizations)
    b = rng.normal(loc=100.0, scale=100 * UNCERTAINTY, size=realizations)

    # Create an function evaluator
    evaluator = FunctionEvaluator(function=partial(rosenbrock, a=a, b=b))

    # Create the basic optimizer
    optimizer = BasicOptimizer(config=CONFIG, evaluator=evaluator)

    # Set the reporter callback
    optimizer.set_results_callback(report)

    # Run the optimization
    optimizer.run(INITIAL_VALUES)

    # Check the results
    assert optimizer.results is not None
    assert optimizer.results.functions is not None
    print(f"Optimal variables: {optimizer.results.evaluations.variables}")
    print(f"Optimal objective: {optimizer.results.functions.target_objective}\n")
    assert np.allclose(optimizer.results.functions.target_objective, 0, atol=1e-1)
    assert np.allclose(optimizer.results.evaluations.variables, 1, atol=1e-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("python rosenbrock.py")
    parser.add_argument(
        "--merge",
        action="store_true",
        help="merge the realizations in gradient calculation",
    )
    main(**vars(parser.parse_args()))
