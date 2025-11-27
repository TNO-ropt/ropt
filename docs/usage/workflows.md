# Optimization Workflows

The section [Running a basic optimization task](basic.md) explains how to run a
single optimization using the [`BasicOptimizer`][ropt.workflow.BasicOptimizer]
class. Although this is sufficient for simple optimization tasks, this class may be 
limited for more complex workflows.

The [`ropt.workflow`][] package provides a powerful and flexible framework for
constructing and executing optimization workflows. It is designed to handle both
simple, single-run optimizations and more complex, customized scenarios. The
framework is built upon three key components:

- **[`ComputeStep`][ropt.plugins.compute_step.base.ComputeStep]**: Defines a
  distinct action within the workflow, such as running an optimization algorithm.
- **[`EventHandler`][ropt.plugins.event_handler.base.EventHandler]**: Responds to
  events emitted by `ComputeStep` instances. This allows for real-time
  monitoring, storing results, or triggering custom logic during the workflow.
- **[`Evaluator`][ropt.plugins.evaluator.base.Evaluator]**: Provides a mechanism
  for `ComputeStep` objects to perform function evaluations, such as running
  simulations on a high-performance computing (HPC) cluster.

Compute steps, event handlers, and evaluators are implemented using the
[`plugin`][ropt.plugins] system. These objects are typically created via
helper fuctions:

- [create_compute_step][ropt.workflow.create_compute_step]: Create
  compute steps.
- [create_event_handler][ropt.workflow.create_event_handler]:
  Create event handlers.
- [create_evaluator][ropt.workflow.create_event_handler]: Create
  evaluators.

After creation, compute steps are executed by calling their
[`run`][ropt.plugins.compute_step.base.ComputeStep.run] method. During
execution, compute steps may emit [`events`][ropt.workflow.Event] to
communicate intermediate results. Event handlers can be added to compute steps
using the
[`add_event_handler`][ropt.plugins.compute_step.base.ComputeStep.add_event_handler]
method. Many compute steps, such as those performing optimizations, will
require the repeated evaluation of a function, which is performed by evaluator
objects passed to the step upon creation.

The following example demonstrates how to construct a workflow from these
components. It finds the optimum of the Rosenbrock function by combining an
optimizer `ComputeStep` with a `tracker` to store the best result.

    
```python
"""Example of optimization of a multi-dimensional Rosenbrock test function.

This example demonstrates optimization of the unmodified (deterministic)
multi-dimensional Rosenbrock function. It shows how to write a minimal
configuration and how to run and monitor the optimization using basic workflow
components.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ropt.config import EnOptConfig
from ropt.enums import EventType
from ropt.evaluator import EvaluatorContext, EvaluatorResult
from ropt.results import FunctionResults
from ropt.workflow import (
    Event,
    create_compute_step,
    create_evaluator,
    create_event_handler,
)

DIM = 5
CONFIG: dict[str, Any] = {
    "variables": {
        "variable_count": DIM,
        "perturbation_magnitudes": 1e-6,
    },
}
initial_values = 2 * np.arange(DIM) / DIM + 0.5


def rosenbrock(
    variables: NDArray[np.float64],
    _: EvaluatorContext,
) -> EvaluatorResult:
    """Function evaluator for the multi-dimensional rosenbrock function.

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
        event: The event to process.
    """
    for item in event.data["results"]:
        if isinstance(item, FunctionResults) and item.functions is not None:
            print(f"  variables: {item.evaluations.variables}")
            print(f"  objective: {item.functions.weighted_objective}\n")


def run_optimization(config: dict[str, Any]) -> FunctionResults:
    """Run the optimization.

    Args:
        config: The configuration of the optimizer.

    Returns:
        The optimal results.
    """
    evaluator = create_evaluator("function_evaluator", callback=rosenbrock)
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
    assert np.allclose(optimal_result.functions.weighted_objective, 0, atol=1e-4)
    assert np.allclose(optimal_result.evaluations.variables, 1, atol=1e-2)


if __name__ == "__main__":
    main()
```
