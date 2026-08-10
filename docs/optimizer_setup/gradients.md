# Stochastic Gradients

Many `ropt` optimizers are gradient-based, but the objective functions are
often black-box simulations with no analytic derivative. `ropt` estimates
gradients stochastically using the *Stochastic Simplex Approximate Gradient*
(StoSAG) approach: the gradient at the current point is approximated from
function values evaluated at the current point and at a number of randomly
perturbed points.

This page explains how perturbations, samplers, function estimators, and the
gradient configuration work together.

## The pieces

| Piece                                                                 | Purpose                                                                       |
| --------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| [`GradientConfig`][ropt.config.GradientConfig]                        | Number of perturbations, success thresholds, evaluation policy.               |
| [`VariablesConfig`][ropt.config.VariablesConfig] perturbation fields  | Magnitudes, scaling type, boundary handling, sampler indices.                 |
| [`SamplerConfig`][ropt.config.SamplerConfig] + samplers tuple         | How perturbation samples are drawn (uniform, normal, Sobol, ...).             |
| [`FunctionEstimatorConfig`][ropt.config.FunctionEstimatorConfig]      | How per-realization gradients combine into the final estimate.                |

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

## Choosing `number_of_perturbations`

More perturbations → more accurate gradient estimates but more evaluator calls
per iteration. With `merge_realizations=False` (the default) the per-realization
gradient is estimated from `number_of_perturbations` samples *per realization*;
with `merge_realizations=True` all realizations are pooled before estimation,
which lets you use a much smaller `number_of_perturbations` (down to 1) at the
cost of losing per-realization signal.

## Tolerating failures

If an evaluator returns NaN for some perturbed rows (e.g. simulator
crashes), the gradient can still be computed when at least
`perturbation_min_success` rows succeeded. Combine with the `realization_min_success` field of
[`RealizationsConfig`][ropt.config.RealizationsConfig]
to also tolerate failed realizations.

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

A sampler draws the perturbation samples used for gradient estimation. The
default [`SciPySampler`][ropt.sampler.scipy.SciPySampler] draws from a standard
normal distribution $N(0, 1)$; other samplers can be added through the plugin
system. Configure the `samplers` tuple at the top level of the config and
reference them by index from `variables.samplers` (see
[Configuration](configuration.md) on index sharing).

### Sample scaling

Samplers produce **unscaled** perturbations — values with a characteristic
magnitude of approximately one. During gradient estimation, these samples are
multiplied element-wise by the `perturbation_magnitudes` defined in
[`VariablesConfig`][ropt.config.VariablesConfig]. This separation means that
`perturbation_magnitudes` directly controls the effective size of the
perturbations, regardless of which sampler is used.

### Shared perturbations

By default, each realization receives its own independently drawn set of
perturbations. Setting the `shared` flag to `True` in
[`SamplerConfig`][ropt.config.SamplerConfig] causes the same perturbation
values to be reused across all realizations. This can reduce noise in the
gradient estimate when the objective function varies smoothly across
realizations.

## Function estimators

A function estimator decides how the gradient samples (and the function
samples for non-default estimators) are combined into the final estimate.
The default
[`DefaultFunctionEstimator`][ropt.plugins.function_estimator.default.DefaultFunctionEstimator]
uses the mean of the functions and gradients. Alternative estimators are configured via the
`function_estimators` tuple and selected per-objective in
[`ObjectiveFunctionsConfig.function_estimators`][ropt.config.ObjectiveFunctionsConfig].

## Where to next

- Filter out unhelpful realizations before estimation:
  [Realization Filters](realization_filters.md).
- Inspect the gradient values produced:
  [Working with Results](results.md) (`GradientResults`).
- Algorithm-specific gradient details: see the relevant backend page under
  [Reference / Optimizer Backends](../reference/backend.md).
