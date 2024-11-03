"""Example of optimization of a multi-dimensional Rosenbrock test function.

This example demonstrates optimization of the unmodified (deterministic)
multi-dimensional Rosenbrock function. It shows how to write a minimal
configuration and how to run and monitor the optimization.
"""

from typing import Any, Dict

import numpy as np
from numpy.typing import NDArray

from ropt.enums import EventType
from ropt.evaluator import EvaluatorContext, EvaluatorResult
from ropt.plan import Event, OptimizationPlanRunner
from ropt.results import FunctionResults

DIM = 5

CONFIG: Dict[str, Any] = {
    "variables": {
        "initial_values": 2 * np.arange(DIM) / DIM + 0.5,
    },
    "gradient": {
        "perturbation_magnitudes": 1e-6,
    },
}


def rosenbrock(variables: NDArray[np.float64], _: EvaluatorContext) -> EvaluatorResult:
    """Function evaluator for the 4D rosenbrock function.

    This function returns a tuple containing the calculated objectives and
    `None`, the latter because no constraints are calculated.

    Args:
        variables: The variables to evaluate.
        dimension: The number of variables.

    Returns:
        The calculated objective, and `None`
    """
    objectives = np.zeros((variables.shape[0], 1), dtype=np.float64)
    for v_idx in range(variables.shape[0]):
        for d_idx in range(DIM - 1):
            x, y = variables[v_idx, d_idx : d_idx + 2]
            objectives[v_idx, 0] += (1.0 - x) ** 2 + 100 * (y - x * x) ** 2
    return EvaluatorResult(objectives=objectives)


def report(event: Event) -> None:
    """Report results of an evaluation.

    Args:
        event: event data.
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
        config: The configuration of the optimizer.

    Returns:
        The optimal results.
    """
    optimal_result = (
        OptimizationPlanRunner(config, rosenbrock)
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


def main() -> None:
    """Run the example and check the result."""
    optimal_result = run_optimization(CONFIG)
    assert optimal_result is not None
    assert optimal_result.functions is not None
    assert np.allclose(optimal_result.functions.weighted_objective, 0, atol=1e-4)
    assert np.allclose(optimal_result.evaluations.variables, 1, atol=1e-2)


if __name__ == "__main__":
    main()
