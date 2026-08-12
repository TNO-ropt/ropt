# Mixed-Integer Optimization

The full script is
[examples/simple/discrete.py](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/discrete.py).
It solves a small integer problem with the gradient-free *differential evolution*
method.

## Declare integer variables and the method

Variable types go in the config. `VariableType.INTEGER` marks the variables as
integer-valued. The `backend` section chooses the differential evolution method:

```python
from ropt.enums import VariableType

config = {
    "variables": {
        "variable_count": 2,
        "types": VariableType.INTEGER,
        "lower_bounds": [0.0, 0.0],
        "upper_bounds": [10.0, 10.0],
    },
    "optimizer": {"max_functions": 5},
    "backend": {
        "method": "differential_evolution",
        "options": {"rng": 4},
        "parallel": False,
    },
}
```

`VariableType` comes from `ropt.enums`, not from `ropt.simple`.

## Add the constraint

The problem keeps `x + y` at or below `10`. Declare the bound in the config under
`nonlinear_constraints`, and have the evaluation function return the constraint
value after the objective:

```python
config["nonlinear_constraints"] = {
    "lower_bounds": [-np.inf],
    "upper_bounds": [10.0],
}


def function(variables, context):
    x, y = variables
    objective = -min(3.0 * x, y)
    return [float(objective), float(x + y)]
```

As in [Constrained Optimization](constrained.md), the config declares the bounds
while the objective returns the value; `ropt` pairs them by position (objectives
first, then constraints).

## Run it

```python
from ropt.simple import optimize

result = optimize(config, INITIAL_VALUES, function, report=report)
```

The best point is `[3, 7]`.

## Next

- The full simple API: [Running Optimizations](../running/running.md).
- Variable settings: [Configuration](../optimizer_setup/configuration.md).
