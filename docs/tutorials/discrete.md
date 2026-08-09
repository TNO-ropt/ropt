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

The problem keeps `x + y` at or below `10`. Here it is a nonlinear constraint, so
the objective returns the objective and the constraint:

```python
def function(variables, context):
    x, y = variables
    objective = -min(3.0 * x, y)
    return [float(objective), float(x + y)]
```

## Run it

```python
from ropt.simple import optimize

result = optimize(config, INITIAL_VALUES, function, report=report)
```

The best point is `[3, 7]`.

## Next

- The full simple API: [The Simple API](../usage/simple.md).
- Variable settings: [Configuration](../usage/configuration.md).
