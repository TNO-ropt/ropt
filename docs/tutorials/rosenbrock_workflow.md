# Using the Workflow Framework

This tutorial demonstrates optimization of the multi-dimensional Rosenbrock
function using the workflow framework directly. This approach offers more
control and flexibility compared to
[`BasicOptimizer`][ropt.workflow.BasicOptimizer].

!!! tip "Source Code"
    The complete source code for this tutorial is available at
    [examples/rosenbrock_workflow.py](https://github.com/TNO-ropt/ropt/blob/main/examples/rosenbrock_workflow.py).


## When to Use the Workflow Framework

Use the workflow framework when you need:

- Custom event handling beyond simple callbacks
- Chained or nested optimizations
- Fine-grained control over the optimization process
- Access to all events emitted during optimization

For simple single-run optimizations,
[`BasicOptimizer`][ropt.workflow.BasicOptimizer] is usually sufficient.


## Imports and Constants

```python
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
from ropt.workflow.event_handlers import CallbackHandler, ResultHandler

DIM = 5
CONFIG: dict[str, Any] = {
    "variables": {
        "variable_count": DIM,
        "perturbation_magnitudes": 1e-6,
    },
}
INITIAL_VALUES = 2 * np.arange(DIM) / DIM + 0.5
UNCERTAINTY = 0.1
```

Note the additional imports for the workflow framework:

- [`EnOptContext`][ropt.context.EnOptContext] — The optimization context
- [`EnOptEventType`][ropt.enums.EnOptEventType] — Event type enumeration
- [`EnOptEvent`][ropt.events.EnOptEvent] — Event objects
- [`OptimizationStep`][ropt.workflow.compute_steps.OptimizationStep] — The compute step
- [`BatchEvaluator`][ropt.workflow.evaluators.BatchEvaluator] — Wraps a batch callback
- [`CallbackHandler`][ropt.workflow.event_handlers.CallbackHandler],
  [`ResultHandler`][ropt.workflow.event_handlers.ResultHandler] — Event handlers


## The Batch Evaluation Callback

The evaluation callback is the same as in the basic tutorial:

```python
def rosenbrock(
    variables: NDArray[np.float64],
    context: EvaluationBatchContext,
    a: NDArray[np.float64],
    b: NDArray[np.float64],
) -> EvaluationBatchResult:
    objectives = np.zeros((variables.shape[0], 1), dtype=np.float64)
    for v_idx, r in enumerate(context.realizations):
        for d_idx in range(DIM - 1):
            x, y = variables[v_idx, d_idx : d_idx + 2]
            objectives[v_idx, 0] += (a[r] - x) ** 2 + b[r] * (y - x * x) ** 2
    return EvaluationBatchResult(objectives=objectives)
```


## Event-Based Progress Reporting

With the workflow framework, the report callback receives
[`EnOptEvent`][ropt.events.EnOptEvent] objects instead of raw results:

```python
def report(event: EnOptEvent) -> None:
    for item in event.results:
        if isinstance(item, FunctionResults) and item.functions is not None:
            print(f"  variables: {item.evaluations.variables}")
            print(f"  objective: {item.functions.target_objective}\n")
```

This callback filters for FunctionResults and prints the current best variables
and objective value. The event object provides access to:

- `event.results` — The results tuple
- `event.event_type` — The type of event ([`EnOptEventType`][ropt.enums.EnOptEventType])
- `event.context` — The optimization context


## Running the Optimization

The main function shows how to build and run a workflow:

```python
def main(*, merge: bool = False) -> None:
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
    result_handler = ResultHandler()
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
    print(f"Optimal variables: {optimal_result.evaluations.variables}")
    print(f"Optimal objective: {optimal_result.functions.target_objective}\n")
```

The workflow approach involves these steps:

1. **Create a batch evaluator**: Wrap the callback in a
   [`BatchEvaluator`][ropt.workflow.evaluators.BatchEvaluator]

2. **Create an optimization step**: The
   [`OptimizationStep`][ropt.workflow.compute_steps.OptimizationStep] runs
   the optimization algorithm

3. **Add a result handler**: The
   [`ResultHandler`][ropt.workflow.event_handlers.ResultHandler] stores the
   best result found

4. **Add a callback handler**: The
   [`CallbackHandler`][ropt.workflow.event_handlers.CallbackHandler] calls
   our report function for `FINISHED_EVALUATION` events

5. **Create the context**: Parse the config into an
   [`EnOptContext`][ropt.context.EnOptContext]

6. **Run the step**: Execute with initial variables and context

7. **Retrieve results**: Get the best result from the handler


## Command-Line Interface

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser("python rosenbrock.py")
    parser.add_argument(
        "--merge",
        action="store_true",
        help="merge the realizations in gradient calculation",
    )
    main(**vars(parser.parse_args()))
```


## Running the Example

```bash
# Default: 10 realizations with 5 perturbations
python rosenbrock_workflow.py

# Use merged realizations
python rosenbrock_workflow.py --merge
```


## Comparison with BasicOptimizer

| Aspect | BasicOptimizer | Workflow Framework |
|--------|----------------|-------------------|
| Setup | Minimal | More explicit |
| Event handling | Callback only | Full event system |
| Result access | `optimizer.results` | Via `ResultHandler` |
| Flexibility | Limited | High |
| Best for | Simple optimizations | Complex workflows |


## Next Steps

- [Basic Optimization Tutorial](rosenbrock_basic.md) — Use BasicOptimizer for
  simpler setup
- [Function Evaluator Tutorial](rosenbrock_function.md) — Use per-evaluation
  callbacks
- [Optimization Workflows](../usage/workflows.md) — Full reference on the
  workflow framework, event handlers, and evaluators
- [Working with Results](../usage/results.md) — Understanding result objects
