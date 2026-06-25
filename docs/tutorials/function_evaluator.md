# Using FunctionEvaluator

This tutorial demonstrates optimization of the multi-dimensional Rosenbrock
function using [`FunctionEvaluator`][ropt.workflow.evaluators.FunctionEvaluator].
This approach uses a simpler per-evaluation callback instead of handling batches
directly.

!!! tip "Source Code"
    The complete source code for this tutorial is available at
    [examples/function_evaluator.py](https://github.com/TNO-ropt/ropt/blob/main/examples/function_evaluator.py).


## When to Use FunctionEvaluator

Use [`FunctionEvaluator`][ropt.workflow.evaluators.FunctionEvaluator] when:

- Your evaluation logic is simpler to express for a single variable vector
- You don't need to vectorize across evaluations
- You're prototyping and want the simplest possible callback

The trade-off is that your callback is called once per evaluation, rather than
once per batch.


## Imports and Constants

```python
import argparse
from functools import partial
from typing import Any

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

from ropt.results import FunctionResults, Results
from ropt.workflow import BasicOptimizer
from ropt.workflow.evaluators import (
    EvaluationFunctionContext,
    EvaluationFunctionResult,
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
```

Note that we import [`FunctionEvaluator`][ropt.workflow.evaluators.FunctionEvaluator]
together with the helper types
[`EvaluationFunctionContext`][ropt.workflow.evaluators.EvaluationFunctionContext]
and
[`EvaluationFunctionResult`][ropt.workflow.evaluators.EvaluationFunctionResult]
instead of the batch evaluation types.


## The Function Callback

A **function callback** handles a single evaluation at a time:

```python
def rosenbrock(
    variables: NDArray[np.float64],
    context: EvaluationFunctionContext,
    a: NDArray[np.float64],
    b: NDArray[np.float64],
) -> EvaluationFunctionResult:
    objective = 0.0
    for idx in range(DIM - 1):
        x, y = variables[idx : idx + 2]
        r = context.realization
        objective += (a[r] - x) ** 2 + b[r] * (y - x * x) ** 2
    return EvaluationFunctionResult(objectives=objective)
```

Key differences from the batch callback:

- `variables` is a **1-D array** (single variable vector)
- The realization index is accessed via `context.realization`
- Additional metadata is available on `context`: `perturbation`, `batch_id`,
  and `eval_idx`
- Returns an
  [`EvaluationFunctionResult`][ropt.workflow.evaluators.EvaluationFunctionResult]
  carrying the objective values (and optional constraints and
  `evaluation_info`) for this single evaluation


## Progress Reporting

The report callback is identical to the basic tutorial:

```python
def report(results: tuple[Results, ...]) -> None:
    for item in results:
        if isinstance(item, FunctionResults) and item.functions is not None:
            print(f"  variables: {item.evaluations.variables}")
            print(f"  objective: {item.functions.target_objective}\n")
```

This callback filters for FunctionResults and prints the current best variables and objective value.


## Running the Optimization

The main difference is creating a [`FunctionEvaluator`][ropt.workflow.evaluators.FunctionEvaluator]
to wrap the function callback:

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

    # Create an function evaluator
    evaluator = FunctionEvaluator(function=partial(rosenbrock, a=a, b=b))

    # Create the basic optimizer
    optimizer = BasicOptimizer(config=CONFIG, evaluator=evaluator)

    # Set the reporter callback
    optimizer.set_results_callback(report)

    # Run the optimization
    optimizer.run(INITIAL_VALUES)

    # Report the results
    print(f"Optimal variables: {optimizer.results.evaluations.variables}")
    print(f"Optimal objective: {optimizer.results.functions.target_objective}\n")
```

The key step is:

```python
evaluator = FunctionEvaluator(function=partial(rosenbrock, a=a, b=b))
```

This wraps the per-evaluation function callback in a
[`FunctionEvaluator`][ropt.workflow.evaluators.FunctionEvaluator], which handles
calling it for each evaluation in a batch.


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
python function_evaluator.py

# Use merged realizations
python function_evaluator.py --merge
```


## Comparison with Batch Callback

| Aspect | Batch Callback | Function Callback |
|--------|---------------|-------------------|
| Input | 2-D array (all evaluations) | 1-D array (single evaluation) |
| Realization | Via `context.realizations` | Via `context.realization` |
| Return type | `EvaluationBatchResult` | `EvaluationFunctionResult` |
| Vectorization | Possible | Not applicable |
| Simplicity | More complex | Simpler |
| Best for | Performance-critical code | Prototyping, simple logic |


## Next Steps

- [Ensemble-based Optimization](ensemble.md) — Use batch callbacks for
  better performance
- [Using the Workflow Framework](workflow.md) — Use the workflow framework for
  more control
- [Writing Evaluation Callbacks](../usage/evaluation_callbacks.md) — Detailed
  reference on evaluation callbacks and `FunctionEvaluator`
