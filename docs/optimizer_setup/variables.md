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
perturbed, and is reported back unchanged. This is the way to explore a subset
of a larger problem without rewriting the configuration around it.

## Integer variables

The `types` field makes variables integer-valued rather than continuous. That
choice constrains the rest of the configuration, because an integer variable
cannot be differentiated — see
[Discrete and Mixed-Integer Variables](discrete.md).

## Scaling

Variables reach the optimizer as $y = (x - o)/s$, using the `scales` and
`offsets` fields, and are reported back as $x = s\,y + o$. Both directions come
from the same two arrays, so they cannot disagree.

This matters when variables differ by orders of magnitude. An optimizer takes a
step of the same size in every direction and judges convergence with a single
tolerance for all of them, and neither is meaningful unless the variables are
comparable in size. The bounds are one source of scales, and
[`scales_and_offsets_from_bounds`][ropt.utils.scales_and_offsets_from_bounds]
derives them:

```python
from ropt.utils import scales_and_offsets_from_bounds

scales, offsets = scales_and_offsets_from_bounds([0.0, 100.0], [1.0, 600.0])
```

That maps both variables onto $[0, 1]$. Scales must be positive, since a scale
is a change of units and nothing else. There is no `auto_scale` for variables:
the bounds are the only information available before the run starts, and using
them is a choice rather than a default.

Everything describing the variables moves with them — the bounds, the absolute
perturbation magnitudes, and the linear constraints — so scaling is not
something that has to be applied consistently by hand. The details are under
[Scaling the variables](configuration_sections.md#variable-scales).

!!! warning "Scaling changes what a run reports"

    Results carry both domains: `variables` in the units configured here, and
    `scaled.variables` in the optimizer's. Only the first is comparable between
    runs that scale differently. The runnable demonstration is
    [examples/simple/scaling.py](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/scaling.py),
    and [Working with Results](../running/results.md#scaling-of-results)
    describes which fields have two domains.

## Perturbations

The remaining fields of the section — `perturbation_magnitudes`,
`perturbation_types`, `boundary_types`, `samplers` and `seed` — describe how
variables are perturbed when a gradient is estimated, rather than the variables
themselves. They are covered in
[Stochastic Gradients](gradients.md) and [Samplers](samplers.md).
