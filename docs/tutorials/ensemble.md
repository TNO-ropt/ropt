# Ensemble Optimization

The full script is
[examples/simple/ensemble.py](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/ensemble.py).
It minimizes the Rosenbrock function whose parameters are uncertain: they differ
across a set of *realizations*.

## Add realizations to the config

The `realizations` section lists a weight per realization. The `gradient` section
controls how many perturbations `ropt` uses to estimate the gradient:

```python
config = {
    "variables": {"variable_count": DIM, "perturbation_magnitudes": 1e-6},
    "realizations": {"weights": [1.0] * realizations},
    "gradient": {"number_of_perturbations": 5},
}
```

## Use the realization number in the objective

The objective receives a `context`. Its `realization` field tells you which
realization to compute, so you can pick the right uncertain parameters:

```python
def rosenbrock(variables, context):
    r = context.realization
    objective = 0.0
    for d_idx in range(DIM - 1):
        x, y = variables[d_idx : d_idx + 2]
        objective += (a[r] - x) ** 2 + b[r] * (y - x * x) ** 2
    return float(objective)
```

Here `a` and `b` are arrays of uncertain parameters, one value per realization.

## Run it

`optimize` minimizes the weighted average objective over all realizations:

```python
from ropt.simple import optimize

result = optimize(config, INITIAL_VALUES, rosenbrock, report=report)
```

The result still converges close to all ones.

The script also accepts `--merge`, which estimates the gradient from a single
perturbation per realization (`merge_realizations`) instead of several
perturbations each. See [Stochastic Gradients](../optimizer_setup/gradients.md)
for the trade-off.

## Next

- Add constraints: [Constrained Optimization](constrained.md).
- More on realizations: [Configuration](../optimizer_setup/configuration.md).
