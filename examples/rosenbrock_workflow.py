"""Example of optimization of a multi-dimensional Rosenbrock test function.

This example demonstrates optimization of the a modified multi-dimensional
Rosenbrock function that exhibits uncertainty in its parameters. It shows how to
write a minimal configuration and how to run and monitor the optimization.

This script demonstrate the use of a custom workflow to run the optimization
instead of a basic optimizer.

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

from ropt.context import EnOptContext
from ropt.enums import EnOptEventType
from ropt.evaluation import (
    EvaluationBatchContext,
    EvaluationBatchResult,
)
from ropt.events import EnOptEvent
from ropt.results import FunctionResults
from ropt.workflow.compute_steps import OptimizationStep
from ropt.workflow.evaluators import BatchEvaluator
from ropt.workflow.event_handlers import CallbackHandler, ResultsHandler

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
    context: EvaluationBatchContext,
    a: NDArray[np.float64],
    b: NDArray[np.float64],
) -> EvaluationBatchResult:
    """Batch evaluation callback for the multi-dimensional rosenbrock function.

    Args:
        variables: The variables to evaluate.
        context:   Evaluator context.
        a:         The 'a' parameters.
        b:         The 'b' parameters.

    Returns:
        An `EvaluationBatchResult` object containing the calculated objectives.
    """
    objectives = np.zeros((variables.shape[0], 1), dtype=np.float64)
    for v_idx, r in enumerate(context.realizations):
        for d_idx in range(DIM - 1):
            x, y = variables[v_idx, d_idx : d_idx + 2]
            objectives[v_idx, 0] += (a[r] - x) ** 2 + b[r] * (y - x * x) ** 2
    return EvaluationBatchResult(objectives=objectives)


def report(event: EnOptEvent) -> None:
    """Report results of an evaluation.

    Args:
        event: The event to process.
    """
    for item in event.results:
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

    # Create a batch evaluator
    evaluator = BatchEvaluator(callback=partial(rosenbrock, a=a, b=b))

    # Create an optimization step
    step = OptimizationStep(evaluator=evaluator)

    # Add a result handler to track the best result
    result_handler = ResultsHandler()
    step.add_event_handler(result_handler)

    # Add an event handler to report results after each evaluation
    reporter = CallbackHandler(
        callback=report, event_types={EnOptEventType.FINISHED_EVALUATION}
    )
    step.add_event_handler(reporter)

    # Create an optimization context from the configuration
    context = EnOptContext.model_validate(CONFIG)

    # Run the optimization step using the initial values
    step.run(variables=INITIAL_VALUES, context=context)

    # Retrieve the best result from the result handler
    optimal_result = result_handler["results"]

    # Check the results
    assert optimal_result is not None
    assert optimal_result.functions is not None
    print(f"Optimal variables: {optimal_result.evaluations.variables}")
    print(f"Optimal objective: {optimal_result.functions.target_objective}\n")
    assert np.allclose(optimal_result.functions.target_objective, 0, atol=1e-1)
    assert np.allclose(optimal_result.evaluations.variables, 1, atol=1e-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("python rosenbrock.py")
    parser.add_argument(
        "--merge",
        action="store_true",
        help="merge the realizations in gradient calculation",
    )
    main(**vars(parser.parse_args()))
