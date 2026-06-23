# Stochastic Rosenbrock Optimization

This tutorial extends the [deterministic Rosenbrock example](rosenbrock_deterministic.md)
by introducing **uncertainty** in the function parameters. It demonstrates
ensemble-based optimization using [`BasicOptimizer`][ropt.workflow.BasicOptimizer]
with a batch evaluation callback.

!!! tip "Source Code"
    The complete source code for this tutorial is available at
    [examples/rosenbrock_basic.py](https://github.com/TNO-ropt/ropt/blob/main/examples/rosenbrock_basic.py).


## Adding Uncertainty

In the [deterministic tutorial](rosenbrock_deterministic.md), we used fixed
parameters $a = 1$ and $b = 100$. Here, we introduce **uncertainty** by sampling
these parameters from normal distributions for each realization:

- $a \sim \mathcal{N}(1.0, 0.1)$
- $b \sim \mathcal{N}(100, 10)$

This makes the problem stochastic: the optimizer must find variable values that
perform well across all realizations, not just for a single set of parameters.


## Imports and Constants

First, let's set up the imports and define constants:

```python
import argparse
from functools import partial
from typing import Any

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

from ropt.evaluation import (
    EvaluationBatchContext,
    EvaluationBatchResult,
)
from ropt.results import FunctionResults, Results
from ropt.workflow import BasicOptimizer

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

The configuration dictionary specifies only the essential parameters:

- `variable_count`: We optimize 5 variables
- `perturbation_magnitudes`: Small perturbations for numerical gradient estimation
- We set a relative uncertainty of 0.1


## The Batch Evaluation Callback

A **batch evaluation callback** receives all variable vectors that need
evaluation at once. This is efficient when you can vectorize computations or
dispatch all evaluations in parallel:

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

Key points:

- `variables` is a 2-D array with shape `(n_evaluations, n_variables)`
- `context.realizations` maps each row to a realization index
- The realization index is used to pick a value for `a` and `b`, which creates
  variation in the ensemble of functions
- Returns an [`EvaluationBatchResult`][ropt.evaluation.EvaluationBatchResult]
  with the objective values


## Progress Reporting

We define a callback to report results after each evaluation:

```python
def report(results: tuple[Results, ...]) -> None:
    for item in results:
        if isinstance(item, FunctionResults) and item.functions is not None:
            print(f"  variables: {item.evaluations.variables}")
            print(f"  objective: {item.functions.target_objective}\n")
```

This callback filters for [`FunctionResults`][ropt.results.FunctionResults] and
prints the current best variables and objective value.


## Running the Optimization

The main function sets up and runs the optimization:

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

    # Create the basic optimizer
    optimizer = BasicOptimizer(config=CONFIG, evaluator=partial(rosenbrock, a=a, b=b))

    # Set the reporter callback
    optimizer.set_results_callback(report)

    # Run the optimization
    optimizer.run(INITIAL_VALUES)

    # Report the results
    print(f"Optimal variables: {optimizer.results.evaluations.variables}")
    print(f"Optimal objective: {optimizer.results.functions.target_objective}\n")
```

The main steps are:

1. **Configure realizations and gradients**: The `--merge` flag controls whether
   to merge realizations in gradient calculation (see below)
2. **Generate uncertain parameters**: Random `a` and `b` values for each realization
3. **Create the optimizer**: Pass the config and evaluation callback (using
   `functools.partial` to bind the parameters)
4. **Set a results callback**: For progress reporting
5. **Run the optimization**: Starting from `INITIAL_VALUES`
6. **Check the results**: Verify the optimizer found a good solution


## The `merge_realizations` Option

The `--merge` flag demonstrates an important gradient calculation option. When
`merge_realizations` is `True`:

- Gradient contributions from all realizations are merged
- Fewer perturbations are needed (1 instead of 5)
- More realizations can be used efficiently (50 instead of 10)

This is useful when evaluations are expensive and you want to maximize the
information extracted from each perturbation. See
[Stochastic Gradients](../usage/gradients.md) for more details.


## Command-Line Interface

The script uses `argparse` to provide command-line options:

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
python rosenbrock_basic.py

# Use merged realizations: 50 realizations with 1 perturbation
python rosenbrock_basic.py --merge
```

Both variants produce optimal variables near $(1, 1, 1, 1, 1)$ with an objective
value near $0$.


## Next Steps

- [Function Evaluator Tutorial](rosenbrock_function.md) — Use a simpler
  per-evaluation callback
- [Workflow Tutorial](rosenbrock_workflow.md) — Use the workflow framework for
  more control
- [Basic Optimization](../usage/basic.md) — Detailed explanation of
  `BasicOptimizer` and its API
- [Writing Evaluation Callbacks](../usage/evaluation_callbacks.md) — More on
  batch callbacks and the `FunctionEvaluator` alternative
- [Working with Results](../usage/results.md) — Understanding result objects
  and how to inspect them
