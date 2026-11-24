# Finite-Difference Gradients

!!! note
    This page covers the finite-difference methods of the built-in gradient
    estimator. The default method, which estimates gradients from random
    perturbations, is described in [Stochastic Gradients](gradients.md).

A finite-difference gradient changes one variable at a time by a fixed step,
and divides the change in each function by that step. The built-in gradient
estimator provides three variants, selected by the `method` field of the
`gradient` section:

| Method     | Points per variable | Derivative of variable $i$                  |
| ---------- | ------------------- | ------------------------------------------- |
| `forward`  | 1                   | $\dfrac{f(x + h_i e_i) - f(x)}{h_i}$         |
| `backward` | 1                   | $\dfrac{f(x) - f(x - h_i e_i)}{h_i}$         |
| `central`  | 2                   | $\dfrac{f(x + h_i e_i) - f(x - h_i e_i)}{2 h_i}$ |

Here $e_i$ is the unit vector of variable $i$, and $h_i$ is its step. The error
of the forward and backward differences is proportional to $h_i$, and that of
the central difference to $h_i^2$. Forward and backward differences are exact
for linear functions, and central differences for quadratic functions. Central
differences need twice as many evaluations.

```python
CONFIG = {
    "variables": {
        "variable_count": 3,
        "perturbation_magnitudes": 1e-4,
    },
    "gradient": {"method": "central"},
}
```

The method may also be written with the plugin name, as `"default/central"`.

## Steps and evaluations

The step $h_i$ of each variable is its `perturbation_magnitudes` value, in the
configured variable units. [Variable scaling](variables.md#scaling) divides the
steps by `variables.scales`, as it does for stochastic gradients.

Only the free variables, selected by `variables.mask`, are changed. For each
realization, a gradient needs one evaluation per free variable with `forward` or
`backward`, and up to two with `central`, in addition to the evaluation at the
current point. All realizations are evaluated at the same points.

The gradient is computed from the points that were evaluated, which differ from
the table above where a bound limits the step.

## Bounds

The points stay within the variable bounds, and `boundary_types` is not used:

- If the step crosses a bound, the step is taken in the other direction: a
  forward difference becomes a backward difference, and the reverse.
- If the step does not fit on both sides of the current point, a central
  difference becomes a forward or backward difference, using one point. That
  point is chosen as for a forward difference.
- If the distance between the bounds is smaller than the step, the step is
  reduced to the largest one that stays within the bounds.

## Realizations and failures

A gradient is computed for each realization, and the gradients are combined
with the [function estimators](function_estimators.md), as for stochastic
gradients. The gradients of the individual realizations are stored in the
`estimates` field of [`GradientResults`][ropt.results.GradientResults].

A realization fails when any of its evaluations fails. The
`realization_min_success` field of
[`RealizationsConfig`][ropt.config.RealizationsConfig] sets how many
realizations must succeed.

## Settings that do not apply

The finite-difference methods have no options. The `gradient` fields
`number_of_perturbations`, `perturbation_min_success` and `merge_realizations`
apply only to the stochastic method, and setting any of them with a
finite-difference method raises an error. The samplers and the seed are not
used.

[Initial gradient results](../running/initial_values.md) passed as `g0` must
contain finite-difference points, each changing a single variable. Results of a
stochastic gradient raise a `ValueError` when the gradient is computed.
