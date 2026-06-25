"""Example of optimization of a multi-dimensional Rosenbrock test function.

This example demonstrates optimization of the a modified multi-dimensional
Rosenbrock function that exhibits uncertainty in its parameters. It shows how to
write a minimal configuration and how to run and monitor the optimization.

We slightly modify the Rosenbrock function to have two real and two discrete
variables, with values 1, 2, 3, and 4. The optimization will be performed using
a differential evolution algorithm, which is suitable for mixed-integer
optimization problems.
"""

from functools import partial
from typing import Any

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

from ropt.context import EnOptContext
from ropt.enums import EnOptEventType, VariableType
from ropt.events import EnOptEvent
from ropt.results import FunctionResults
from ropt.workflow.compute_steps import OptimizationStep
from ropt.workflow.evaluators import (
    EvaluatorFunctionContext,
    EvaluatorFunctionResult,
    FunctionEvaluator,
)
from ropt.workflow.event_handlers import CallbackHandler, ResultsHandler

DIM = 4
REALIZATIONS = 10
CONFIG: dict[str, Any] = {
    "variables": {
        "variable_count": DIM,
        "perturbation_magnitudes": 1e-6,
        "lower_bounds": 0.0,
        "upper_bounds": 10.0,
        "mask": [True, True, True, True],
        "types": [
            VariableType.REAL,
            VariableType.REAL,
            VariableType.INTEGER,
            VariableType.INTEGER,
        ],
    },
    "backend": {
        "method": "differential_evolution",
        "options": {"rng": 4},
        "max_iterations": 50,
    },
    "realizations": {
        "weights": [1.0] * REALIZATIONS,
    },
}
INITIAL_VALUES = [1.0, 1.0, 1.0, 1.0]
UNCERTAINTY = 0.1


def rosenbrock(
    variables: NDArray[np.float64],
    context: EvaluatorFunctionContext,
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
    scaled = variables / np.arange(1, DIM + 1)
    for idx in range(DIM - 1):
        x, y = scaled[idx : idx + 2]
        r = context.realization
        objective += (a[r] - x) ** 2 + b[r] * (y - x * x) ** 2
    return EvaluatorFunctionResult(objectives=objective)


def report(event: EnOptEvent) -> None:
    """Report results of an evaluation.

    Args:
        event: The event to process.
    """
    for item in event.results:
        if isinstance(item, FunctionResults) and item.functions is not None:
            print(f"batch: {item.batch_id}")
            print(f"  variables: {item.evaluations.variables}")
            print(f"  objective: {item.functions.target_objective}\n")


def main() -> None:
    """Run the example and check the result."""
    # Generate random parameters for the Rosenbrock function
    rng = default_rng(seed=123)
    a = rng.normal(loc=1.0, scale=UNCERTAINTY, size=REALIZATIONS)
    b = rng.normal(loc=100.0, scale=100 * UNCERTAINTY, size=REALIZATIONS)

    # Create a batch evaluator
    evaluator = FunctionEvaluator(function=partial(rosenbrock, a=a, b=b))

    # Create an optimization step
    optimizer = OptimizationStep(evaluator=evaluator)

    # Add a result handler to track the best result
    result_handler = ResultsHandler()
    optimizer.add_event_handler(result_handler)

    # Add an event handler to report results after each evaluation
    reporter = CallbackHandler(
        callback=report, event_types={EnOptEventType.FINISHED_EVALUATION}
    )
    optimizer.add_event_handler(reporter)

    # Create an optimization context from the configuration
    context = EnOptContext.model_validate(CONFIG)

    # Run the optimization step using the initial values
    optimizer.run(variables=INITIAL_VALUES, context=context)

    # Retrieve the best result from the result handler
    optimal_result = result_handler["results"]

    # Check the results
    assert optimal_result is not None
    assert optimal_result.functions is not None
    print(f"Optimal batch: {optimal_result.batch_id}")
    print(f"Optimal variables: {optimal_result.evaluations.variables}")
    print(f"Optimal objective: {optimal_result.functions.target_objective}\n")
    assert np.allclose(optimal_result.functions.target_objective, 0, atol=1e-1)
    assert np.allclose(optimal_result.evaluations.variables, [1, 2, 3, 4], atol=1e-1)


if __name__ == "__main__":
    main()
