# Quickstart

This page shows the smallest complete `ropt` program: it minimizes a simple
function.

## Install `ropt`

```bash
pip install ropt
```

See [Installation](installation.md) for optional extras.

## A minimal optimization

```python
import numpy as np

from ropt.simple import optimize

# 1. Describe the problem: three variables.
config = {
    "variables": {
        "variable_count": 3,
        "perturbation_magnitudes": 1e-6,
    },
}


# 2. The objective: the value to minimize.
def objective(variables, context):
    # `context` identifies which evaluation this is; not needed here.
    return float(np.sum((variables - 1.0) ** 2))


# 3. Run the optimization from a starting point equal to zero.
result = optimize(config, np.zeros(3), objective)

print(f"best variables: {result.variables}")
print(f"best objective: {result.target_objective}")
```

Running this finds variables close to `[1, 1, 1]`.

## How it works

Every `ropt` optimization needs three things:

1. **A config dictionary** — it describes the problem. Here we set only the
   minimum: how many variables there are, and a small `perturbation_magnitudes`
   value that `ropt` uses to estimate gradients. See
   [Configuration](../optimizer_setup/configuration.md) for the full list of settings.
2. **An evaluation function** — a Python function that takes a set of variable
   values and returns the number to minimize. See
   [Running Optimizations](../running/running.md#the-evaluation-function).
3. **A start point** — the variable values to start from.

[`optimize`][ropt.simple.optimize] combines these three, runs the optimization,
and returns an [`OptimizeResult`][ropt.simple.OptimizeResult] with the best
values it found.

## Where to next

- A fuller walkthrough: [Deterministic Optimization](deterministic.md).
- The complete simple API: [Running Optimizations](../running/running.md).
- All configuration settings: [Configuration](../optimizer_setup/configuration.md).
