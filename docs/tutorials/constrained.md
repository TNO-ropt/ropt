# Constrained Optimization

The full script is
[examples/simple/constrained.py](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/constrained.py).
It adds a nonlinear constraint to the ensemble Rosenbrock problem.

## Declare the constraint bounds

Nonlinear constraint bounds go in the config. Here the constraint value must stay
at or below `-1`:

```python
config = {
    "variables": {"variable_count": DIM, "perturbation_magnitudes": 1e-6, ...},
    "realizations": {"weights": [1.0] * REALIZATIONS},
    "nonlinear_constraints": {"lower_bounds": -np.inf, "upper_bounds": -1.0},
}
```

## Return the constraint from the objective

With one objective and one nonlinear constraint, the objective returns a list:
the objective first, then the constraint value:

```python
def rosenbrock(variables, context):
    r = context.realization
    objective = ...          # the Rosenbrock value
    x, y = variables[:2]
    constraint = (x - a[r]) ** 3 - y
    return [float(objective), float(constraint)]
```

## Flag violations in the report

The full result is on `result.results`. Its `constraint_info` reports how much
each constraint is violated:

```python
def report(result):
    info = result.results.constraint_info
    if (
        info is not None
        and info.nonlinear_violation is not None
        and np.any(info.nonlinear_violation > 0)
    ):
        print(f"  constraint violation: {info.nonlinear_violation}")
```

## Run it

`constraint_tolerance` sets how close a constraint must be to count as satisfied:

```python
from ropt.simple import optimize

result = optimize(
    config, INITIAL_VALUES, rosenbrock, report=report, constraint_tolerance=1e-6
)
```

The script also accepts `--linear` to add a deterministic linear constraint,
declared entirely in the config.

## Next

- Integer variables: [Mixed-Integer Optimization](discrete.md).
- All constraint settings:
  [Configuration](../optimizer_configuration/configuration.md).
