# Your First Optimization

The full script for this tutorial is
[examples/simple/optimize.py](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/optimize.py).
It minimizes the Rosenbrock function, whose minimum is at all ones.

## Describe the problem

The config dictionary lists the variables. `perturbation_magnitudes` is the small
step `ropt` uses to estimate gradients:

```python
DIM = 5
CONFIG = {
    "variables": {
        "variable_count": DIM,
        "perturbation_magnitudes": 1e-6,
    },
}
```

## Write the objective

The objective takes one set of variable values and returns the number to
minimize:

```python
def rosenbrock(variables, context):
    objective = 0.0
    for d_idx in range(DIM - 1):
        x, y = variables[d_idx : d_idx + 2]
        objective += (1.0 - x) ** 2 + 100 * (y - x * x) ** 2
    return float(objective)
```

## Report progress (optional)

A `report` callback is called after every evaluation:

```python
def report(result):
    if result.target_objective is not None:
        print(f"  objective: {result.target_objective}")
```

## Run it and read the result

```python
from ropt.simple import optimize

result = optimize(CONFIG, INITIAL_VALUES, rosenbrock, report=report)
print(f"optimal variables: {result.variables}")
print(f"optimal objective: {result.target_objective}")
```

`result.variables` holds the best variables found — close to `[1, 1, 1, 1, 1]`.

## Next

- The full simple API: [Running Optimizations](../running/running.md).
- Optimize over uncertain realizations: [Ensemble Optimization](ensemble.md).
