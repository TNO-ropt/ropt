"""Example of optimization of a multi-dimensional Rosenbrock test function.

This example demonstrates optimization of the a modified multi-dimensional
Rosenbrock function that exhibits uncertainty in its parameters. It shows how to
write a minimal configuration and how to run and monitor the optimization using
basic workflow components.
"""

from functools import partial
from typing import Any

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

from ropt.config import EnOptConfig
from ropt.enums import EventType
from ropt.results import FunctionResults
from ropt.workflow import (
    Event,
    create_compute_step,
    create_evaluator,
    create_event_handler,
)

DIM = 5
UNCERTAINTY = 0.1
CONFIG: dict[str, Any] = {
    "variables": {
        "variable_count": DIM,
        "perturbation_magnitudes": 1e-6,
    },
    "realizations": {
        "weights": [1.0] * 10,
    },
    "gradient": {
        "number_of_perturbations": 5,
    },
}
initial_values = 2 * np.arange(DIM) / DIM + 0.5


def rosenbrock(
    variables: NDArray[np.float64],
    realization: int,
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    **kwargs: Any,  # noqa: ANN401, ARG001
) -> NDArray[np.float64]:
    """Function evaluator for the multi-dimensional rosenbrock function.

    Args:
        variables:    The variables to evaluate.
        realization:  Realization number.
        a:            The 'a' parameters.
        b:            The 'b' parameters.
        kwargs:       Unused keyword arguments.

    Returns:
        The calculated objective.
    """
    objective = 0.0
    for d_idx in range(DIM - 1):
        x, y = variables[d_idx : d_idx + 2]
        objective += (a[realization] - x) ** 2 + b[realization] * (y - x * x) ** 2
    return np.asarray([objective])


def report(event: Event) -> None:
    """Report results of an evaluation.

    Args:
        event: The event to process.
    """
    for item in event.data["results"]:
        if isinstance(item, FunctionResults) and item.functions is not None:
            print(f"  variables: {item.evaluations.variables}")
            print(f"  objective: {item.functions.weighted_objective}\n")


def run_optimization(config: dict[str, Any]) -> FunctionResults:
    """Run the optimization.

    Args:
        config:    The configuration of the optimizer.

    Returns:
        The optimal results.
    """
    rng = default_rng(seed=123)

    realizations = len(config["realizations"]["weights"])
    a = rng.normal(loc=1.0, scale=UNCERTAINTY, size=realizations)
    b = rng.normal(loc=100.0, scale=100 * UNCERTAINTY, size=realizations)

    evaluator = create_evaluator(
        "function_evaluator", function=partial(rosenbrock, a=a, b=b)
    )
    step = create_compute_step("optimizer", evaluator=evaluator)

    tracker = create_event_handler("tracker")
    step.add_event_handler(tracker)

    reporter = create_event_handler(
        "observer", callback=report, event_types={EventType.FINISHED_EVALUATION}
    )
    step.add_event_handler(reporter)

    step.run(variables=initial_values, config=EnOptConfig.model_validate(config))

    optimal_result: FunctionResults = tracker["results"]
    assert optimal_result is not None
    assert optimal_result.functions is not None

    print(f"Optimal variables: {optimal_result.evaluations.variables}")
    print(f"Optimal objective: {optimal_result.functions.weighted_objective}\n")

    return optimal_result


def main() -> None:
    """Run the example and check the result."""
    optimal_result = run_optimization(CONFIG)
    assert optimal_result is not None
    assert optimal_result.functions is not None
    assert np.allclose(optimal_result.functions.weighted_objective, 0, atol=1e-1)
    assert np.allclose(optimal_result.evaluations.variables, 1, atol=1e-1)


if __name__ == "__main__":
    main()
