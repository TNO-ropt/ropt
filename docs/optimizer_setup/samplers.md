# Samplers

A sampler draws the perturbations that [stochastic gradient
estimation](gradients.md) evaluates around the current point. Which sampler is
used decides *where* those points are placed; how far they are placed is decided
separately, by `perturbation_magnitudes`.

The default is [`SciPySampler`][ropt.sampler.scipy.SciPySampler], drawing from a
standard normal distribution $N(0, 1)$. It also offers uniform and truncated
normal distributions, and the quasi-random Sobol, Halton and Latin hypercube
sequences, which cover a space more evenly than independent draws.

## Selecting a sampler

Samplers are listed at the top level of the configuration and referenced from
`variables.samplers`, one entry per variable, so different variables can be
perturbed by different samplers:

```python
CONFIG = {
    "variables": {
        "variable_count": 4,
        "perturbation_magnitudes": 1e-6,
        "samplers": [0, 0, 1, 1],   # variables 0,1 use sampler 0; 2,3 use sampler 1
    },
    "samplers": [
        {"method": "scipy/default"},   # index 0
        {"method": "scipy/sobol"},     # index 1
    ],
}
```

Omit both fields entirely to perturb every variable with the default sampler.
See [Sharing optimizer components by
key](configuration.md#sharing-optimizer-components-by-key) for the indexing
pattern, which objectives and constraints use in the same way.

## Sample scaling

Samplers produce **unscaled** perturbations — values with a characteristic
magnitude of approximately one. During gradient estimation these samples are
multiplied element-wise by the `perturbation_magnitudes` defined in
[`VariablesConfig`][ropt.config.VariablesConfig].

The separation keeps the two independent: the sampler determines the
distribution of the perturbed points, and `perturbation_magnitudes` determines
their distance from the current point, whichever sampler is in use.

## Shared perturbations

By default each realization receives its own independently drawn set of
perturbations. Setting the `shared` flag to `True` in
[`SamplerConfig`][ropt.config.SamplerConfig] reuses the same perturbation values
across all realizations:

```python
"samplers": [{"method": "scipy/default", "shared": True}]
```

This can reduce noise in the gradient estimate when the objective varies
smoothly across realizations, because the realizations then differ only in the
objective and not also in where it was sampled. The flag has no effect on a
sampler whose values do not depend on the realization in the first place.

## Writing a custom sampler

A custom sampler is a plugin implementing the [`Sampler`][ropt.sampler.Sampler]
base class, whose docstring documents the shape the samples must have and how
the `mask` and `shared` settings affect them. Like the other component kinds, a
sampler defined in a script or a notebook is made selectable with
[`register_plugin`][ropt.plugins.register_plugin].

The runnable script is
[examples/simple/sampler.py](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/sampler.py),
which perturbs one variable at a time. With one perturbation per variable and a
unit step, the gradient estimate is a forward finite difference — which shows
that perturbations need not be random at all. Its second method scales each step
by a random factor instead, and the two runs differ in `shared` to show where
that flag changes the samples.
