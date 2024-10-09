"""Example of optimization of a multi-dimensional Rosenbrock test function.

This example demonstrates optimization of the a modified multi-dimensional
Rosenbrock function that exhibits uncertainty in its parameters. It shows how to
write a minimal configuration and how to run and monitor the optimization.
"""

import sys
from functools import partial
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

from ropt.enums import EventType
from ropt.evaluator import EvaluatorContext, EvaluatorResult
from ropt.plan import Event, OptimizationPlanRunner
from ropt.results import FunctionResults

DIM = 5
UNCERTAINTY = 0.1


CONFIG: Dict[str, Any] = {
    "variables": {
        "initial_values": 2 * np.arange(DIM) / DIM + 0.5,
    },
    "gradient": {
        "perturbation_magnitudes": 1e-6,
    },
}


def rosenbrock(
    variables: NDArray[np.float64],
    context: EvaluatorContext,
    parameters: NDArray[np.float64],
) -> EvaluatorResult:
    """Function evaluator for the rosenbrock function.

    This function returns a tuple containing the calculated objectives and
    `None`, the latter because no constraints are calculated.

    Args:
        variables:  The variables to evaluate
        context:    Evaluator context
        parameters: The set of parameters for all realizations

    Returns:
        The calculated objective, and `None`
    """
    objectives = np.zeros((variables.shape[0], 1), dtype=np.float64)
    a, b = parameters[:2, :]
    for v_idx, r in enumerate(context.realizations):
        for d_idx in range(DIM - 1):
            x, y = variables[v_idx, d_idx : d_idx + 2]
            objectives[v_idx, 0] += (a[r] - x) ** 2 + b[r] * (y - x * x) ** 2
    return EvaluatorResult(objectives=objectives)


def report(event: Event) -> None:
    """Report results of an evaluation.

    Args:
        event: event data
    """
    assert event.results is not None
    for item in event.results:
        if isinstance(item, FunctionResults) and item.functions is not None:
            print(f"result: {item.result_id}")
            print(f"  variables: {item.evaluations.variables}")
            print(f"  objective: {item.functions.weighted_objective}\n")


def run_optimization(config: Dict[str, Any]) -> FunctionResults:
    """Run the optimization.

    Args:
        config: The configuration of the optimizer

    Returns:
        The optimal results.
    """
    rng = default_rng(seed=123)

    realizations = len(config["realizations"]["weights"])
    parameters = np.zeros((4, realizations), dtype=np.float64)
    parameters[0, :] = rng.normal(loc=1.0, scale=UNCERTAINTY, size=realizations)
    parameters[1, :] = rng.normal(loc=100.0, scale=100 * UNCERTAINTY, size=realizations)

    optimal_result = (
        OptimizationPlanRunner(CONFIG, partial(rosenbrock, parameters=parameters))
        .add_observer(EventType.FINISHED_EVALUATION, report)
        .run()
        .results
    )
    assert optimal_result is not None
    assert optimal_result.functions is not None

    print(f"BEST RESULT: {optimal_result.result_id}")
    print(f"  variables: {optimal_result.evaluations.variables}")
    print(f"  objective: {optimal_result.functions.weighted_objective}\n")

    return optimal_result


def main(argv: Optional[List[str]] = None) -> None:
    """Run the example and check the result.

    Args:
        argv: Optional command line arguments.
    """
    if argv is not None and "--merge" in argv:
        realizations = 50
        CONFIG["realizations"] = {"weights": [1.0] * realizations}
        CONFIG["gradient"]["number_of_perturbations"] = 1
        CONFIG["gradient"]["merge_realizations"] = True
    else:
        realizations = 10
        CONFIG["realizations"] = {"weights": [1.0] * realizations}
        CONFIG["gradient"]["number_of_perturbations"] = 5

    optimal_result = run_optimization(CONFIG)
    assert optimal_result is not None
    assert optimal_result.functions is not None
    assert np.allclose(optimal_result.functions.weighted_objective, 0, atol=1e-1)
    assert np.allclose(optimal_result.evaluations.variables, 1, atol=1e-1)


if __name__ == "__main__":
    main(sys.argv)
