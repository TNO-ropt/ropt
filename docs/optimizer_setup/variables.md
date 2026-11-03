# Variables

The variable vector is the one thing every configuration must define: the values
the optimizer is allowed to change. `variable_count` sets how many there are,
and everything else about them — bounds, types, which are held fixed, and the
units the optimizer works in — is optional.

```python
CONFIG = {
    "variables": {
        "variable_count": 3,
        "lower_bounds": -1.0,
        "upper_bounds": [1.0, 2.0, 3.0],
    },
}
```

A single value is broadcast to every variable, so `lower_bounds` above applies
to all three. The defaults are $-\infty$ and $+\infty$, which is to say
unbounded. The field-by-field description is under
[`variables`](configuration_sections.md#variables).

## Holding variables fixed

The `mask` field marks which variables the optimizer may change. It takes one
boolean per variable, `True` for free and `False` for fixed, and defaults to all
free:

```python
"variables": {
    "variable_count": 3,
    "mask": [True, True, False],   # the third variable is held at its start value
}
```

A fixed variable keeps the value it had in the initial vector, is never
perturbed, and is reported back unchanged. This lets you explore a subset of a
larger problem without rewriting the configuration around it. It is also how
[nested optimizations](../running/nested.md#splitting-the-variables) assign
different variables to the outer and inner optimizers.

## Integer variables

The `types` field makes variables integer-valued rather than continuous. Set it
to `"integer"` to make every variable integer-valued, or give one type per
variable to mix integer and continuous variables:

```python
"variables": {
    "variable_count": 4,
    "types": ["real", "real", "integer", "integer"],
}
```

An evaluation function receives integer variables as floats with integral
values. In the built-in SciPy backend, only `differential_evolution` honors
integer types; other SciPy methods silently treat them as continuous. This
method also requires bounds on every variable. See
[Choosing an Optimizer](optimizer.md#gradient-based-or-gradient-free) before
selecting a method, and the runnable
[all-integer](https://github.com/TNO-ropt/ropt/blob/main/examples/discrete.py)
and
[mixed-variable](https://github.com/TNO-ropt/ropt/blob/main/examples/mixed.py)
examples.

## Scaling

Variables reach the optimizer as $y = (x - o)/s$, using the `scales` and
`offsets` fields, and are reported back as $x = s\,y + o$. Both directions come
from the same two arrays, so they cannot disagree.

Whether scaling helps depends on the optimizer. It can be useful when variables
differ greatly in magnitude, and the bounds are one possible source of scales.
[`scales_and_offsets_from_bounds`][ropt.utils.scales_and_offsets_from_bounds]
derives them:

```python
from ropt.utils import scales_and_offsets_from_bounds

scales, offsets = scales_and_offsets_from_bounds([0.0, 100.0], [1.0, 600.0])
```

That maps both variables onto $[0, 1]$. Scales must be positive, and there is no
`auto_scale` for variables. The bounds and linear constraints are transformed
along with the variables. Perturbation magnitudes are specified in the configured
variable units and divided by the variable scales.
See [Scaling the variables](configuration_sections.md#variable-scales) for the
transformation details.

!!! warning "Scaling changes what an optimization reports"

    Results expose `variables` in the configured units and `scaled.variables`
    in the optimizer's units. Compare `variables` between optimizations that use
    different scales. The runnable demonstration is
    [examples/scaling.py](https://github.com/TNO-ropt/ropt/blob/main/examples/scaling.py),
    and [Working with Results](../results/results.md#scaling-of-results)
    describes which fields have two domains.

    [`ConstraintInfo`][ropt.results.ConstraintInfo] reports bound distances in
    configured units. The corresponding `scaled.constraint_info` fields give
    the distances in the optimizer's units; offsets shift variables and bounds
    together and do not change those distances.

## Perturbations

The remaining fields of the section — `perturbation_magnitudes`,
`boundary_types`, `samplers` and `seed` — describe how variables are perturbed
when a gradient is estimated, rather than the variables themselves. They are
covered in [Stochastic Gradients](gradients.md) and [Samplers](samplers.md).
