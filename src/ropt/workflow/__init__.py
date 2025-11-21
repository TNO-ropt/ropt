r"""Optimization workflow functionality.

The `ropt.workflow` package provides a powerful and flexible framework for
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

Example:
    ````python
    import numpy as np
    from numpy.typing import NDArray

    from ropt.config import EnOptConfig
    from ropt.evaluator import EvaluatorContext, EvaluatorResult
    from ropt.workflow import (
        create_compute_step,
        create_evaluator,
        create_event_handler,
    )

    DIM = 5
    CONFIG = {
        "variables": {
            "variable_count": DIM,
            "perturbation_magnitudes": 1e-6,
        },
    }
    initial_values = 2 * np.arange(DIM) / DIM + 0.5


    def rosenbrock(variables: NDArray[np.float64], _: EvaluatorContext) -> EvaluatorResult:
        objectives = np.zeros((variables.shape[0], 1), dtype=np.float64)
        for v_idx in range(variables.shape[0]):
            for d_idx in range(DIM - 1):
                x, y = variables[v_idx, d_idx : d_idx + 2]
                objectives[v_idx, 0] += (1.0 - x) ** 2 + 100 * (y - x * x) ** 2
        return EvaluatorResult(objectives=objectives)


    evaluator = create_evaluator("function_evaluator", callback=rosenbrock)
    step = create_compute_step("optimizer", evaluator=evaluator)
    tracker = create_event_handler("tracker")
    step.add_event_handler(tracker)
    step.run(variables=initial_values, config=EnOptConfig.model_validate(CONFIG))

    print(f"Optimal variables: {tracker['results'].evaluations.variables}")
    print(f"Optimal objective: {tracker['results'].functions.weighted_objective}")
    ````

Note:
    This example demonstrates the manual construction of a workflow, which is
    ideal for complex, multi-step processes. For straightforward, single-run
    optimizations, the [`BasicOptimizer`][ropt.workflow.BasicOptimizer] class
    offers a simpler, high-level interface.
"""

from ._basic_optimizer import BasicOptimizer
from ._events import Event
from ._factory import create_compute_step, create_evaluator, create_event_handler

__all__ = [
    "BasicOptimizer",
    "Event",
    "create_compute_step",
    "create_evaluator",
    "create_event_handler",
]
