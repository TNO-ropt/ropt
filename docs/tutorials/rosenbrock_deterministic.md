# Deterministic Rosenbrock Optimization

This tutorial demonstrates optimization of the classic multi-dimensional
Rosenbrock function using [`BasicOptimizer`][ropt.workflow.BasicOptimizer]. This
is the simplest possible optimization example in `ropt`.

!!! tip "Source Code"
    The complete source code for this tutorial is available at
    [examples/rosenbrock_deterministic.py](https://github.com/TNO-ropt/ropt/blob/main/examples/rosenbrock_deterministic.py).


## The Rosenbrock Function

The [Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function) is
a classic test problem for optimization algorithms. In its multi-dimensional
form:

$$
f(\mathbf{x}) = \sum_{i=1}^{n-1} \left[ (a - x_i)^2 + b(x_{i+1} - x_i^2)^2 \right]
$$

The global minimum is at $\mathbf{x} = (a, a, \ldots, a)$ where $f(\mathbf{x}) = 0$.
For the standard case with $a = 1$ and $b = 100$, the minimum is at
$\mathbf{x} = (1, 1, \ldots, 1)$.

This tutorial uses the deterministic version with fixed $a = 1$ and $b = 100$.
The subsequent tutorials introduce uncertainty by sampling these parameters.


## Imports and Constants

```python
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ropt.evaluation import EvaluationBatchContext, EvaluationBatchResult
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
```

The configuration dictionary specifies only the essential parameters:

- `variable_count`: We optimize 5 variables
- `perturbation_magnitudes`: Small perturbations for numerical gradient estimation


## The Evaluation Callback

The evaluation callback computes the Rosenbrock function for each variable
vector in the batch:

```python
def rosenbrock(
    variables: NDArray[np.float64], _: EvaluationBatchContext
) -> EvaluationBatchResult:
    objectives = np.zeros((variables.shape[0], 1), dtype=np.float64)
    for v_idx in range(variables.shape[0]):
        for d_idx in range(DIM - 1):
            x, y = variables[v_idx, d_idx : d_idx + 2]
            objectives[v_idx, 0] += (1.0 - x) ** 2 + 100 * (y - x * x) ** 2
    return EvaluationBatchResult(objectives=objectives)
```

Key points:

- `variables` is a 2-D array with shape `(n_evaluations, n_variables)`
- The context argument is unused in this deterministic case
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

This callback filters for FunctionResults and prints the current best variables and objective value.


## Running the Optimization

The main function creates the optimizer and runs it:

```python
def main() -> None:
     # Create the basic optimizer
    optimizer = BasicOptimizer(CONFIG, rosenbrock)

    # Set the reporter callback
    optimizer.set_results_callback(report)

    # Run the optimization
    optimizer.run(INITIAL_VALUES)

    # Report the results
    print(f"Optimal variables: {optimizer.results.evaluations.variables}")
    print(f"Optimal objective: {optimizer.results.functions.target_objective}\n")
```

The steps are:

1. **Create the optimizer**: Pass the config and evaluation callback
2. **Set a results callback**: For progress reporting
3. **Run the optimization**: Starting from `INITIAL_VALUES`
4. **Check the results**: Verify the optimizer found the global minimum


## Entry Point

```python
if __name__ == "__main__":
    main()
```


## Running the Example

```bash
python rosenbrock_deterministic.py
```

The optimizer finds variables near $(1, 1, 1, 1, 1)$ with an objective value
near $0$.


## Next Steps

- [Stochastic Optimization](rosenbrock_basic.md) — Add uncertainty to the
  problem with multiple realizations
- [Function Evaluator](rosenbrock_function.md) — Use per-evaluation callbacks
- [Workflow Framework](rosenbrock_workflow.md) — Use the workflow framework for
  more control
