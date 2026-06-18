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

from ropt.context import EnOptContext
from ropt.enums import EnOptEventType
from ropt.events import EnOptEvent
from ropt.results import FunctionResults
from ropt.workflow.compute_steps import EnsembleOptimizer
from ropt.workflow.evaluators import FunctionEvaluator
from ropt.workflow.event_handlers import Observer, Tracker

DIM = 5
UNCERTAINTY = 0.1
REALIZATIONS = 10

CONFIG: dict[str, Any] = {
    "variables": {
        "variable_count": DIM,
        "perturbation_magnitudes": 1e-6,
    },
    "realizations": {
        "weights": [1.0] * REALIZATIONS,
    },
    "gradient": {
        "number_of_perturbations": 5,
    },
}
INITIAL_VALUES = 2 * np.arange(DIM) / DIM + 0.5


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


def report(event: EnOptEvent) -> None:
    """Report results of an evaluation.

    Args:
        event: The event to process.
    """
    for item in event.results:
        if isinstance(item, FunctionResults) and item.functions is not None:
            print(f"  variables: {item.evaluations.variables}")
            print(f"  objective: {item.functions.target_objective}\n")


def main() -> None:
    """Run the example and check the result."""
    rng = default_rng(seed=123)
    a = rng.normal(loc=1.0, scale=UNCERTAINTY, size=REALIZATIONS)
    b = rng.normal(loc=100.0, scale=100 * UNCERTAINTY, size=REALIZATIONS)

    evaluator = FunctionEvaluator(function=partial(rosenbrock, a=a, b=b))
    step = EnsembleOptimizer(evaluator=evaluator)

    tracker = Tracker()
    step.add_event_handler(tracker)

    reporter = Observer(
        callback=report, event_types={EnOptEventType.FINISHED_EVALUATION}
    )
    step.add_event_handler(reporter)

    step.run(variables=INITIAL_VALUES, context=EnOptContext.model_validate(CONFIG))

    optimal_result: FunctionResults = tracker["results"]
    assert optimal_result is not None
    assert optimal_result.functions is not None

    print(f"Optimal variables: {optimal_result.evaluations.variables}")
    print(f"Optimal objective: {optimal_result.functions.target_objective}\n")

    assert np.allclose(optimal_result.functions.target_objective, 0, atol=1e-1)
    assert np.allclose(optimal_result.evaluations.variables, 1, atol=1e-1)


if __name__ == "__main__":
    main()
