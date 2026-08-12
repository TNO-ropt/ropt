# Deterministic Optimization

This page walks through a complete deterministic optimization with the simple
API, one step at a time. The full runnable script is
[examples/simple/optimize.py](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/optimize.py).

We minimize the Rosenbrock function in five dimensions; its minimum is at all
ones. The objective is *deterministic*: the same variables always produce the
same value. [Ensemble-Based Optimization](ensemble.md) covers the case where the
objective is uncertain.

## 1. Describe the problem

The config dictionary describes the optimization. For a plain problem you only
need the variables:

```python
DIM = 5
config = {
    "variables": {
        "variable_count": DIM,
        "perturbation_magnitudes": 1e-6,
    },
}
```

`variable_count` is how many variables there are. `perturbation_magnitudes` is a
small step size that `ropt` uses to estimate gradients. Every other setting keeps
its default; see [Configuration](../optimizer_setup/configuration.md) for the full list.

## 2. Write the evaluation function

The evaluation function is a Python function. It receives one set of variable
values and returns the number to minimize:

```python
import numpy as np


def rosenbrock(variables: np.ndarray, _) -> float:
    objective = 0.0
    for i in range(DIM - 1):
        x, y = variables[i : i + 2]
        objective += (1.0 - x) ** 2 + 100 * (y - x * x) ** 2
    return float(objective)
```

- `variables` is a 1-D array with one value per variable.
- The objective always receives a second argument that identifies the
  evaluation. It is not needed for a deterministic problem, so we ignore it here.

The function returns a single number. You can also return a list (objectives
first, then constraints) or attach metadata; see
[Running Optimizations](../running/running.md#the-evaluation-function).

## 3. Follow the progress (optional)

To watch the run, write a small `report` function. It is called after every
evaluation with an [`EvaluateResult`][ropt.simple.EvaluateResult]:

```python
from ropt.simple import EvaluateResult


def report(result: EvaluateResult) -> None:
    if result.target_objective is not None:
        print(f"  objective: {result.target_objective}")
```

## 4. Run it

Give [`optimize`][ropt.simple.optimize] the config, a start point, the objective,
and (optionally) the report callback:

```python
from ropt.simple import optimize

initial_values = 2 * np.arange(DIM) / DIM + 0.5
result = optimize(config, initial_values, rosenbrock, report=report)
```

This runs the optimization to completion and returns the result.

## 5. Read the result

`optimize` returns an [`OptimizeResult`][ropt.simple.OptimizeResult]:

```python
print(f"exit code:         {result.exit_code}")
print(f"optimal variables: {result.variables}")
print(f"optimal objective: {result.target_objective}")
```

- `result.variables` is the best set of variables found (or `None` if the run
  produced no valid result).
- `result.target_objective` is the objective value there.
- `result.exit_code` says why the run stopped.
- `result.results` holds the full low-level result, if you need every detail.

## Where to next

- Optimizing under uncertainty:
  [Ensemble-Based Optimization](ensemble.md).
- The complete simple API — evaluating, running many optimizations, collecting
  results: [Running Optimizations](../running/running.md).
- Every configuration setting: [Configuration](../optimizer_setup/configuration.md).
- Step-by-step tutorials: [Tutorials](../tutorials/index.md).
