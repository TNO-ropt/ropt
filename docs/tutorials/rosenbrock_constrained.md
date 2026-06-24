# Constrained Optimization

This tutorial demonstrates optimization of the Rosenbrock function with
**constraints**. It shows how to specify bound constraints, nonlinear
constraints and linear constraints.

!!! tip "Source Code"
    The complete source code for this tutorial is available at
    [examples/rosenbrock_constrained.py](https://github.com/TNO-ropt/ropt/blob/main/examples/rosenbrock_constrained.py).


## Types of Constraints

`ropt` supports three types of constraints:

- **Bound constraints**: The value of each variable of the solution can be
  limited by specifying bounds in the configuration.

- **Nonlinear constraints**: Evaluated by your callback function alongside the
  objective. Like objectives, they can be stochastic (varying across
  realizations).

- **Linear constraints**: Specified directly in the configuration as coefficient
  matrices and bounds. These are deterministic and handled directly by the
  optimizer.

!!! note "Constraint violations"
    Specifying constraints does not guarantee that all intermediate evaluations
    will satisfy them. Nonlinear constraints in particular may be violated
    during optimization, depending on the backend used. Always verify constraint
    satisfaction when inspecting results.

## The constrained Rosenbrock function

This tutorial extends the [stochastic Rosenbrock example](rosenbrock_basic.md) by adding non-linear and linear constraints. The non-linear constraint is a function of the first two variables:

$$
(x_1 - a)^3 - x_2 + 1 \le 0
$$

Note that this constraint depends on the $a$ parameter and is therefore also stochastic in nature.

In addition we can add a very simple linear constraints that just eliminates one variable:

$$
x_4 = x_5
$$


## Imports and Constants

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
REALIZATIONS = 10
CONFIG: dict[str, Any] = {
    "variables": {
        "variable_count": DIM,
        "perturbation_magnitudes": 1e-6,
        "lower_bounds": -5.0,
        "upper_bounds": 5.0,
    },
    "realizations": {
        "weights": [1.0] * REALIZATIONS,
    },
    "nonlinear_constraints": {
        "lower_bounds": -np.inf,
        "upper_bounds": -1.0,
    },
}
INITIAL_VALUES = 2 * np.arange(DIM) / DIM + 0.5
UNCERTAINTY = 0.1
```

The key addition is the in `variables` and `nonlinear_constraints` sections. In
both we set lower and upper boundaries, for bound and non-linear constraints,
respectively:

- `lower_bounds`: Lower bounds ($b_l$) for each constraint (use `-np.inf` for no lower bound)
- `upper_bounds`: Upper bounds ($b_u$) for each constraint

The lower bounds must be smaller than the upper bounds: $b_l \le b_u$, but they
may be equal to each other. In that case, they specify an equality constraint:
the constraint function is expected to have the exact value $b_l = b_u$ .

The bound constraints defined in the `variables` section define bounds on the
solution variables: $b_l \le \mathbf{x} \le b_u$. Note that the bounds in this
example are optional: the solution for the Rosenbrock will also be found without
these bounds. However, intermediate solutions may differ and the final solution
might slightly different.

The bounds in the `nonlinear-constraints` section define the limits on the value
of the constraint function $g(\mathbf{x})$: $b_l \leq g(\mathbf{x}) \leq b_u$,
in this case we implement the single constraint $g(\mathbf{x}) \le -1$. Note
that this sets the right-hand-side of the inequality to -1, so the constraint
function that we implement below becomes: $g(\mathbf{x}) = (x_1 - a)^3 - x_2$

The linear constraint is not defined yet, because it is optional; it will be
added later.


## Evaluation Callback with Constraints

The callback now calculates both objectives and constraints:

```python
def rosenbrock(
    variables: NDArray[np.float64],
    context: EvaluationBatchContext,
    a: NDArray[np.float64],
    b: NDArray[np.float64],
) -> EvaluationBatchResult:
    objectives = np.zeros((variables.shape[0], 1), dtype=np.float64)
    constraints = np.zeros((variables.shape[0], 1), dtype=np.float64)
    for v_idx, r in enumerate(context.realizations):
        for d_idx in range(DIM - 1):
            x, y = variables[v_idx, d_idx : d_idx + 2]
            objectives[v_idx, 0] += (a[r] - x) ** 2 + b[r] * (y - x * x) ** 2
        x, y = variables[v_idx, :2]
        constraints[v_idx, 0] += (x - a[r]) ** 3 - y
    return EvaluationBatchResult(objectives=objectives, constraints=constraints)

```

Key points:

- The `constraints` array has shape `(n_evaluations, n_constraints)`
- The constraint is stochastic: it uses the realization-specific parameter `a[r]`
- Return both `objectives` and `constraints` in the
  [`EvaluationBatchResult`][ropt.evaluation.EvaluationBatchResult]


## Reporting Constraint Violations

The report callback monitors for constraint violations during optimization:

```python
def report_violations(results: tuple[Results, ...]) -> None:
    for item in results:
        if isinstance(item, FunctionResults) and item.functions is not None:
            violated = np.any(item.constraint_info.nonlinear_violation > 0)
            if violated:
                print("Constraint violation detected:")
                print(f"  variables:  {item.evaluations.variables}")
                print(f"  objective:  {item.functions.target_objective}")
                print(f"  constraint: {item.functions.constraints}")
                print(f"  violation:  {item.constraint_info.nonlinear_violation}\n")
```

This callback accesses the `constraint_info` attribute, which returns a
[`ConstraintInfo`][ropt.results.ConstraintInfo] object containing violation data
for all constraint types.


## Understanding Constraint Violations

A **violation** represents the absolute value of the amount by which a constraint
bound is exceeded, or zero if the constraint is satisfied. For example, if a
constraint requires $g(\mathbf{x}) \ge 0$ and the actual value is $-0.5$, the
violation is $0.5$.

The [`ConstraintInfo`][ropt.results.ConstraintInfo] class provides violation
data for all three constraint types (bound, linear, and nonlinear). This example
only checks nonlinear violations, but the same approach applies to all types.
See [Working with Results](../usage/results.md) for full details on constraint
differences and violations.

!!! tip "Constraint tolerance"
    The [`BasicOptimizer`][ropt.workflow.BasicOptimizer] accepts a single
    `constraint_tolerance` parameter that applies uniformly to all constraint
    types. This mirrors the behavior of
    [`ResultsHandler`][ropt.workflow.event_handlers.ResultsHandler]. For more
    fine-grained control, `ResultsHandler` also accepts a `filter` callable that
    receives a [`Results`][ropt.results.Results] object and returns `True` to
    keep or `False` to discard. This allows custom per-constraint-type logic or
    any other filtering criterion when building a workflow.

## Running the Optimization

```python
def main(*, linear: bool = False) -> None:
    # Generate random parameters for the Rosenbrock function
    rng = default_rng(seed=123)
    a = rng.normal(loc=1.0, scale=UNCERTAINTY, size=REALIZATIONS)
    b = rng.normal(loc=100.0, scale=100 * UNCERTAINTY, size=REALIZATIONS)

    # If desired, add a linear constraint to the configuration
    if linear:
        CONFIG.update(
            {
                "linear_constraints": {
                    "coefficients": [[0.0, 0.0, 0.0, 1.0, -1.0]],
                    "lower_bounds": 0.0,
                    "upper_bounds": 0.0,
                }
            }
        )

    # Create the basic optimizer, set the constraint tolerance
    optimizer = BasicOptimizer(
        CONFIG,
        partial(rosenbrock, a=a, b=b),
        constraint_tolerance=1e-6,
    )

    # Set the reporter callback
    optimizer.set_results_callback(report_violations)

    # Run the optimization
    optimizer.run(INITIAL_VALUES)

    # Check the results
    print(f"Optimal variables:  {optimizer.results.evaluations.variables}")
    print(f"Optimal objective:  {optimizer.results.functions.target_objective}")
    print(f"Optimal constraint: {optimizer.results.functions.constraints}\n")
```

Notable additions:

- **Linear constraints** (optional): Added via the `linear_constraints` config
  section when `--linear` is passed. 

- **Constraint tolerance**: The `constraint_tolerance` parameter controls how
  strictly constraints must be satisfied while tracking the optimal result. A
  smaller value means stricter enforcement.


## Linear Constraints

Linear constraints are entirely specified in the configuration, there is no need to modify 
the function that calculates objectives and nonlinear constraints:

```python
"linear_constraints": {
    "coefficients": [[0.0, 0.0, 0.0, 1.0, -1.0]],  # coefficients for each variable
    "lower_bounds": 0.0,                           # lower bound(s)
    "upper_bounds": 0.0,                           # upper bound(s)
}
```

The constraints are defined as a linear equation $b_l\leq \mathbf{C} \cdot
\mathbf{x} \leq b_u$. The coefficient matrix has one row for each linear
constraint, and the number of columns is equal to the number of variables. The
lower and upper bounds are vectors of length equal to the number of liner
constraints. So in this example $\mathbf{C}$ is a $1 \times 5$ matrix, and $b_l$
and $b_u$ are vectors of length one: $0 \leq x_4 - x_5 \leq 0$, equivalent to $x_4
= x_5$.


## Command-Line Interface

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser("python rosenbrock_constrained.py")
    parser.add_argument(
        "--linear",
        action="store_true",
        help="add a linear constraint",
    )
    args = parser.parse_args()
    main(linear=args.linear)
```


## Running the Example

```bash
# With nonlinear constraint only
python rosenbrock_constrained.py

# With both nonlinear and linear constraints
python rosenbrock_constrained.py --linear
```

## Summary

This tutorial demonstrated:

1. **Nonlinear constraints**: Evaluated by your callback, can be stochastic
2. **Linear constraints**: Specified in configuration, deterministic
3. **Constraint bounds**: Using `-np.inf` for one-sided constraints
4. **Constraint tolerance**: Controlling how strictly constraints are enforced
5. **Constraint violations**: Monitoring if constraints are violated


## Next Steps

- [Configuration](../usage/configuration.md) — Full configuration reference
  including constraint options
- [Writing Evaluation Callbacks](../usage/evaluation_callbacks.md) — More
  details on returning constraints from callbacks
