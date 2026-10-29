# Quickstart

This page shows the smallest complete `ropt` program. It minimizes the
**Rosenbrock function**, a standard test problem for optimizers:

$$ f(x, y) = (1 - x)^2 + 100 \left( y - x^2 \right)^2 $$

Its minimum is at $x = y = 1$, at the bottom of a long curved valley that is
easy to reach but slow to follow.

## Install `ropt`

```bash
pip install ropt
```

See [Installation](installation.md) for optional extras.

## A minimal optimization

```python
import numpy as np

from ropt import optimize

# 1. Describe the problem: two variables.
config = {
    "variables": {
        "variable_count": 2,
        "perturbation_magnitudes": 1e-6,
    },
}


# 2. The objective: the value to minimize.
def rosenbrock(variables, context):
    # `context` identifies which evaluation this is; not needed here.
    x, y = variables
    return float((1.0 - x) ** 2 + 100 * (y - x * x) ** 2)


# 3. Run the optimization from a starting point equal to zero.
result = optimize(config, np.zeros(2), rosenbrock)

if result.results is not None:
    print(f"best variables: {result.results.variables}")
    print(f"best objective: {result.results.target_objective}")
```

Running this finds variables close to `[1, 1]`.

## How it works

Every `ropt` optimization needs three things:

1. **A config dictionary** — it describes the problem. Here we set only the
   minimum: how many variables there are, and a small `perturbation_magnitudes`
   value that `ropt` uses to estimate gradients. See
   [Optimization Choices](../optimizer_setup/choices.md) surveys the main
   decisions, including scaling, constraints, ensemble treatment and methods.
   See [Configuration Sections](../optimizer_setup/configuration_sections.md)
   for the full list of settings.
2. **An evaluation function** — a Python function that takes a set of variable
   values and returns the number to minimize. See
   [Running Optimizations](../running/running.md#the-evaluation-function).
3. **A start point** — the variable values to start from.

[`optimize`][ropt.optimize] combines these three, runs the optimization, and
returns an [`OptimizationResult`][ropt.OptimizationResult]. Its `results` field
holds the best evaluation of the optimization, as a
[`FunctionResults`][ropt.results.FunctionResults] and is `None` if the
optimization found nothing valid.

### Positions and labels

`ropt` identifies variables by their positions in the vector. In the example,
`x, y = variables` assigns Python names to positions 0 and 1; the start point
and any per-variable configuration values use the same order. When there are
several objectives or constraints, their values likewise correspond to
positions in the configuration. The evaluation function returns objectives
first, then nonlinear constraints.

To label the variables in exported results, add `names` to the configuration
before calling `optimize`:

```python
config["names"] = {"variable": ("x", "y")}
```

These labels do not identify values for the optimizer or change their order.
They appear in exported DataFrames, including tables made by
[`DataFrameHandler`][ropt.DataFrameHandler]; without labels, the tables use
numeric indices. The [`names`](../optimizer_setup/configuration_sections.md#names)
section also supports labels for objectives, constraints and other result
dimensions.
