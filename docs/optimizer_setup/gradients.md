# Stochastic Gradients

Many `ropt` optimizers are gradient-based, but the objective functions are
often black-box simulations with no analytic derivative. `ropt` estimates
gradients stochastically using the *Stochastic Simplex Approximate Gradient*
(StoSAG) approach: the gradient at the current point is approximated from
function values evaluated at the current point and at a number of randomly
perturbed points.

This page explains how perturbations, samplers, function estimators, and the
gradient configuration work together. The runnable script is
[examples/ensemble.py](https://github.com/TNO-ropt/ropt/blob/main/examples/ensemble.py),
which exercises this estimation with only `perturbation_magnitudes` and the
realization weights set, leaving the sampler, the estimator and the remaining
gradient settings at their defaults.

## The pieces

| Piece                                                                 | Purpose                                                                       |
| --------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| [`GradientConfig`][ropt.config.GradientConfig]                        | Number of perturbations, success thresholds, evaluation policy.               |
| [`VariablesConfig`][ropt.config.VariablesConfig] perturbation fields  | Magnitudes, boundary handling, sampler indices.                               |
| [`SamplerConfig`][ropt.config.SamplerConfig] + samplers tuple         | How perturbation samples are drawn (uniform, normal, Sobol, ...).             |
| [`FunctionEstimatorConfig`][ropt.config.FunctionEstimatorConfig]      | How per-realization gradients combine into the final estimate.                |

The fields themselves are described under
[`gradient`](configuration_sections.md#gradient) and
[Variable perturbations](configuration_sections.md#variable-perturbations) in
the Configuration Reference.

## Configuration in one place

```python
CONFIG = {
    "variables": {
        "variable_count": 5,
        "perturbation_magnitudes": 1e-5,
        "boundary_types": "truncate",      # or "none", "mirror"
        "samplers": [0, 0, 1, 1, 1],       # per-variable sampler index
    },
    "realizations": {"weights": [1.0] * 10},
    "gradient": {
        "number_of_perturbations": 5,
        "perturbation_min_success": 3,  # allow some failures
        "merge_realizations": False,    # estimate per-realization then combine
        "evaluation_policy": "auto",    # "speculative" | "separate" | "auto"
    },
    "samplers": [{"method": "scipy/default"}, {"method": "scipy/sobol"}],
}
```

## Perturbation magnitude and scaling

`perturbation_magnitudes` sets the absolute scale of the sample applied to
each variable, in the configured variable units. Each sampler value is
multiplied by its magnitude before being added to the variable. For the default
standard normal sampler, the magnitude is the standard deviation before
boundary correction. [Variable scaling](variables.md#scaling) divides these
magnitudes by `variables.scales`.

To express magnitudes as a fraction of finite bound widths, calculate the
absolute sizes explicitly:

```python
import numpy as np

lower = np.array([0.0, 10.0])
upper = np.array([100.0, 30.0])

config = {
    "variables": {
        "variable_count": 2,
        "lower_bounds": lower,
        "upper_bounds": upper,
        "perturbation_magnitudes": 0.05 * (upper - lower),
    },
}
```

This sets the magnitudes to `[5.0, 1.0]`, or 5% of each variable's bound width.

After a perturbed value is computed, it may fall outside the variable bounds.
`boundary_types` controls the correction strategy:

- `"none"` — leave as-is.
- `"truncate"` — clamp to the nearest bound.
- `"mirror"` — reflect through the violated bound.

These are the values of [`BoundaryType`][ropt.enums.BoundaryType], whose
members may be written in their place.

## Reproducibility

The perturbations are drawn from a generator seeded by `variables.seed`, which
defaults to `1`. Repeating an optimization therefore reproduces it exactly, and
changing the seed draws a different set of perturbations.

A plain integer is enough for a single optimization. The field also accepts a
tuple, and giving a unique identifier as its first element is what keeps
optimizations distinct when they are nested or run concurrently: each one draws
its own perturbations while remaining reproducible on its own.

```python
"variables": {"variable_count": 5, "seed": (run_id, 1)}
```

## Choosing `number_of_perturbations`

More perturbations give more accurate gradient estimates at the cost of more
evaluator calls
per iteration. With `merge_realizations=False` (the default) the per-realization
gradient is estimated from `number_of_perturbations` samples *per realization*;
with `merge_realizations=True` all realizations are pooled before estimation,
which lets you use a much smaller `number_of_perturbations` (down to 1) at the
cost of losing per-realization signal.

## Tolerating failures

If an evaluator returns NaN for some perturbed rows (for example, simulator
crashes), the gradient can still be computed when at least
`perturbation_min_success` rows succeeded. Combine with the `realization_min_success` field of
[`RealizationsConfig`][ropt.config.RealizationsConfig]
to also tolerate failed realizations.

[examples/failures.py](https://github.com/TNO-ropt/ropt/blob/main/examples/failures.py)
runs one problem twice: once with the default, where a single failing
realization ends the optimization with `TOO_FEW_REALIZATIONS` and no result, and
once with `realization_min_success` lowered, where the optimization finishes on
the realizations that did work.

## Evaluation policy

[`GradientConfig.evaluation_policy`][ropt.config.GradientConfig] picks one of:

- `"auto"` — compute objectives and gradients strictly when the backend requests
  them. The default, and the policy that performs the fewest evaluations.
- `"speculative"` — also compute the gradient whenever an objective is
  requested. Improves load balancing on HPC clusters when gradient evaluations
  are likely to be needed soon.
- `"separate"` — never combine function and gradient evaluations into a single
  batch. The functions are evaluated first, so a
  [realization filter](realization_filters.md) can set the weight of some
  realizations to zero before the gradient is evaluated. The gradient evaluation
  then skips those realizations. In a combined batch, the weights are not yet
  known and all realizations are evaluated.

## Samplers

A sampler determines *where* the perturbed points are placed, while
`perturbation_magnitudes` determines how far away they are. The default draws from
a standard normal distribution, and quasi-random alternatives are available.
Selecting one, perturbing different variables differently, sharing samples
across realizations and writing your own are all covered in
[Samplers](samplers.md).

## Function estimators

A function estimator determines *how* the per-realization values and gradients are
combined into the single estimate the optimizer receives. The default is a
weighted mean; alternatives, including a measure of dispersion rather than an
average, are covered in [Function Estimators](function_estimators.md).

## See also

- What `evaluation_policy` means when evaluations are submitted concurrently:
  [Evaluating in Parallel](../running/parallel.md).
- When an optimization ends with `TOO_FEW_REALIZATIONS`, or two optimizations
  disagree:
  [Troubleshooting](../troubleshooting/index.md).
