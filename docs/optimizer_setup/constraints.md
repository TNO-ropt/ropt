# Constraints

A constraint restricts which variable vectors count as a valid solution. `ropt`
distinguishes three kinds, and they differ in *who computes them*:

| Kind | Declared | Computed by |
| --- | --- | --- |
| **Bounds** | `variables.lower_bounds` / `upper_bounds` | nobody — the optimizer never leaves the box |
| **Linear** | `linear_constraints` | `ropt`, from the coefficients you give |
| **Nonlinear** | `nonlinear_constraints` | your evaluation function, once per realization |

Bounds and linear constraints are deterministic: they are fully described by the
configuration, so `ropt` can evaluate them itself. A nonlinear constraint is not
— it is computed alongside the objective, which means it can be as expensive as
the objective and, in an ensemble, can differ between realizations.

The runnable script for this page is
[examples/simple/constrained.py](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/constrained.py).
It adds a nonlinear constraint to the ensemble Rosenbrock problem of
[Ensemble-Based Optimization](../getting_started/ensemble.md).

!!! note

    Not every optimization method supports every kind of constraint. `ropt`
    reports an error when the configured method cannot handle what you declared;
    see [Optimizer Backends](../reference/backend.md) for what each method
    supports.

## Declaring bounds and nonlinear constraints

Both live in the configuration, next to the variables:

```python
--8<-- "examples/simple/constrained.py:config"
```

Bounds keep every variable in $[-5, 5]$. The `nonlinear_constraints` section
declares *how many* nonlinear constraints there are and what range each value
must fall in — here a single constraint that must stay at or below $-1$. Use
`-np.inf` or `np.inf` for a one-sided constraint, and equal lower and upper
bounds for an equality.

The section declares only the bounds; the values themselves come from the
evaluation function. See
[`nonlinear_constraints`](configuration_sections.md#nonlinear_constraints) and
[`variables`](configuration_sections.md#variables) for the full field reference.
Constraint values can also be rescaled before the optimizer sees them; see
[Scaling constraints](configuration_sections.md#constraint-scales).

## Returning a constraint from the evaluation function

An evaluation function that has constraints returns a sequence instead of a
single number: **the objectives first, then the constraints**. With one
objective and one constraint, that is a two-element list:

```python
--8<-- "examples/simple/constrained.py:objective"
```

The order is positional — there are no names — so it must match the order in
which the objectives and constraints are configured. See
[Running Optimizations](../running/running.md#the-evaluation-function) for the
other shapes the return value can take.

Because the constraint is computed per realization and uses `A[r]`, it is
*stochastic*: each realization constrains the problem slightly differently, and
`ropt` combines them the same way it combines the objectives.

## Deciding when a constraint is satisfied

A constraint is rarely met exactly, so [`optimize`][ropt.simple.optimize] takes a
`constraint_tolerance`: a result counts as feasible when no constraint is
violated by more than that amount.

```python
--8<-- "examples/simple/constrained.py:run"
```

`result.results` is only ever the best
**feasible** evaluation; if none satisfied the constraints to within the
tolerance, the run returns `None` instead of a best result. A tolerance that is
too tight is a common reason for an empty result — see
[Common Pitfalls](../troubleshooting/index.md).

To watch feasibility as the run proceeds, read `constraint_info` from the result
the `report` callback receives. Its `nonlinear_violation` is zero where a
constraint is met and positive by the amount it is exceeded:

```python
--8<-- "examples/simple/constrained.py:report"
```

## Adding a linear constraint

A linear constraint is a row of coefficients applied to the variable vector,
with bounds on the result. It never reaches your evaluation function. Equal
lower and upper bounds make it an equality — this one forces the fourth and
fifth variables to be equal:

```python
--8<-- "examples/simple/constrained.py:linear"
```

The script adds it when run with `--linear`. One row per constraint, one
coefficient per variable; see
[`linear_constraints`](configuration_sections.md#linear_constraints) for the field
reference, and [Scaling the
equations](configuration_sections.md#linear-constraint-scales) for rescaling a
row before it reaches the optimizer.
