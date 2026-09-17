# Stochastic Gradients

Many `ropt` optimizers are gradient-based, but the objective functions are
often black-box simulations with no analytic derivative. `ropt` estimates
gradients stochastically using the *Stochastic Simplex Approximate Gradient*
(StoSAG) approach: the gradient at the current point is approximated from
function values evaluated at the current point and at a number of randomly
perturbed points.

This page explains how perturbations, samplers, function estimators, and the
gradient configuration work together. The runnable script is
[examples/simple/ensemble.py](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/ensemble.py),
which exercises this machinery with only `perturbation_magnitudes` and the
realization weights set, leaving the sampler, the estimator and the remaining
gradient settings at their defaults.

## The pieces

| Piece                                                                 | Purpose                                                                       |
| --------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| [`GradientConfig`][ropt.config.GradientConfig]                        | Number of perturbations, success thresholds, evaluation policy.               |
| [`VariablesConfig`][ropt.config.VariablesConfig] perturbation fields  | Magnitudes, scaling type, boundary handling, sampler indices.                 |
| [`SamplerConfig`][ropt.config.SamplerConfig] + samplers tuple         | How perturbation samples are drawn (uniform, normal, Sobol, ...).             |
| [`FunctionEstimatorConfig`][ropt.config.FunctionEstimatorConfig]      | How per-realization gradients combine into the final estimate.                |

The fields themselves are described under
[`gradient`](configuration_sections.md#gradient) and
[Variable perturbations](configuration_sections.md#variable-perturbations) in
the Configuration Reference.

## Configuration in one place

```python
from ropt.enums import BoundaryType, PerturbationType

CONFIG = {
    "variables": {
        "variable_count": 5,
        "perturbation_magnitudes": 1e-5,
        "perturbation_types": PerturbationType.ABSOLUTE,  # or PerturbationType.RELATIVE
        "boundary_types": BoundaryType.TRUNCATE_BOTH,     # see BoundaryType
        "samplers": [0, 0, 1, 1, 1],                      # per-variable sampler index
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

`perturbation_magnitudes` sets the scale of the sample applied to each
variable. The `perturbation_types` field decides how that scale is interpreted:

- [`PerturbationType.ABSOLUTE`][ropt.enums.PerturbationType.ABSOLUTE] — the
  magnitude is added directly to the variable value.
- [`PerturbationType.RELATIVE`][ropt.enums.PerturbationType.RELATIVE] — the
  magnitude is multiplied by `upper_bound - lower_bound` before being applied.
  Requires finite bounds.

See [`PerturbationType`][ropt.enums.PerturbationType].

After a perturbed value is computed, it may fall outside the variable bounds.
`boundary_types` controls the correction strategy:

- [`BoundaryType.NONE`][ropt.enums.BoundaryType.NONE] — leave as-is.
- [`BoundaryType.TRUNCATE_BOTH`][ropt.enums.BoundaryType.TRUNCATE_BOTH] —
  clamp to the nearest bound.
- [`BoundaryType.MIRROR_BOTH`][ropt.enums.BoundaryType.MIRROR_BOTH] — reflect
  through the violated bound.

See [`BoundaryType`][ropt.enums.BoundaryType].

## Reproducibility

The perturbations are drawn from a generator seeded by `variables.seed`, which
defaults to `1`. Repeating a run therefore reproduces it exactly, and changing
the seed draws a different set of perturbations.

A plain integer is enough for a single run. The field also accepts a tuple, and
giving a unique identifier as its first element is what keeps runs distinct when
they are nested or evaluated in parallel: each run draws its own perturbations
while remaining reproducible on its own.

```python
"variables": {"variable_count": 5, "seed": (run_id, 1)}
```

## Choosing `number_of_perturbations`

More perturbations → more accurate gradient estimates but more evaluator calls
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

[examples/simple/failures.py](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/failures.py)
runs one problem twice: once with the default, where a single failing
realization ends the run with `TOO_FEW_REALIZATIONS` and no result, and once
with `realization_min_success` lowered, where the run finishes on the
realizations that did work.

## Evaluation policy

[`GradientConfig.evaluation_policy`][ropt.config.GradientConfig] picks one of:

- `"auto"` — compute objectives and gradients strictly when the backend asks
  for them. Default and most efficient.
- `"speculative"` — also compute the gradient whenever an objective is
  requested. Improves load balancing on HPC clusters when gradient evaluations
  are likely to be needed soon.
- `"separate"` — never combine function and gradient evaluations into a single
  batch. Useful with [realization filters](realization_filters.md) that disable
  realizations and reduce gradient work.

## Samplers

A sampler decides *where* the perturbed points are placed, while
`perturbation_magnitudes` decides how far away they are. The default draws from
a standard normal distribution, and quasi-random alternatives are available.
Selecting one, perturbing different variables differently, sharing samples
across realizations and writing your own are all covered in
[Samplers](samplers.md).

## Function estimators

A function estimator decides *how* the per-realization values and gradients are
combined into the single estimate the optimizer receives. The default is a
weighted mean; alternatives, including a measure of dispersion rather than an
average, are covered in [Function Estimators](function_estimators.md).

## See also

- What `evaluation_policy` means when evaluations are submitted concurrently:
  [Parallel Execution and Many Runs](../running/parallel.md).
- When a run ends with `TOO_FEW_REALIZATIONS`, or two runs disagree:
  [Common Pitfalls](../troubleshooting/index.md).
