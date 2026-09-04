# Realization Filters

A realization filter selects, for each evaluation batch, which realizations
contribute to the combined function or gradient value. Filters enable
risk-aware optimization (for example, focusing on the worst-performing
realizations) and common variance-reduction techniques.

`ropt` ships a **CVaR filter** in the `ropt.realization_filter.default`
module, which selects the realizations contributing to the
Conditional-Value-at-Risk tail. It can be configured for objectives, for
constraints, or both.

## How filters fit in

1. You add filter configurations to the top-level `realization_filters` list.
2. You point objectives (or constraints) at a filter by its index in
   `ObjectiveFunctionsConfig.realization_filters` /
   `NonlinearConstraintsConfig.realization_filters`.
3. At each evaluation, the filter is consulted to compute per-realization
   weights that override the static `realizations.weights`.

See [Configuration](configuration.md) for the index-sharing pattern.

## CVaR example

Optimize the conditional expectation of the worst 30% of 10 realizations:

```python
CONFIG = {
    "variables": {"variable_count": 5, "perturbation_magnitudes": 1e-6},
    "realizations": {"weights": [1.0] * 10},
    "objectives": {
        "weights": [1.0],
        "realization_filters": [0],   # objective uses filter 0
    },
    "realization_filters": [
        {
            "method": "default/cvar-objective",
            "options": {"sort": [0], "percentile": 0.3},
        },
    ],
    "gradient": {"number_of_perturbations": 5},
}
```

See [`CVaRObjectiveOptions`][ropt.realization_filter.default.CVaRObjectiveOptions]
for the parameters. The corresponding constraint variant is
[`CVaRConstraintOptions`][ropt.realization_filter.default.CVaRConstraintOptions].

### How CVaR filters work

The `cvar-objective` method:

1. Computes a weighted sum of the objective values specified by the `sort`
   indices for each realization (using the objective weights from the
   configuration). If a single objective index is given, no weighting is
   applied. Objectives marked in
   [`maximize`](configuration.md#objective-direction) have their sign flipped
   first, per objective, so that the sum ranks realizations the way the
   optimizer would.
2. Conceptually sorts realizations by that value, ascending.
3. Identifies the subset corresponding to the `percentile` worst outcomes
   (highest weighted values).
4. Assigns CVaR-derived weights to those realizations. When the percentile
   boundary falls between two realizations, interpolation produces partial
   weights. All other realizations receive zero.
5. Failed realizations (NaN values) are excluded.

The `cvar-constraint` variant applies CVaR to a single constraint function,
with "worst" defined by constraint type:

- **LE (`<=`):** largest positive values (most violated).
- **GE (`>=`):** smallest negative values (most violated).
- **EQ (`==`):** largest absolute values (furthest from zero).

!!! note
    Realizations reach a filter with their objectives already scaled but not
    yet flipped for direction: the flip belongs to the aggregate, and these are
    per-realization values. The filter applies it itself when ranking.
    Constraint filters need no such step, since a constraint is a bound and has
    no direction.

!!! note "Weight normalization"
    The optimizer normalizes all filter-produced weights to sum to one before
    use, so any non-negative values are permissible.

## Interaction with `evaluation_policy`

Filters that disable some realizations only deliver savings on the gradient
side when the optimizer requests gradients separately from functions. Set
`gradient.evaluation_policy = "separate"` (see
[Stochastic Gradients](gradients.md)) to maximize that benefit.

## Writing a custom filter

Custom filters are plugins implementing the
[`RealizationFilter`][ropt.realization_filter.RealizationFilter] base class,
whose docstring documents the methods to implement. Registering a filter with
the plugin system is only required when it should be selectable via
[`RealizationFilterConfig`][ropt.config.RealizationFilterConfig]; otherwise, an
instance can be passed directly in the `realization_filters` field of
[`EnOptContext`][ropt.context.EnOptContext].

## Where to next

- Combine filters with function estimators:
  [Function Estimators](function_estimators.md).
- Inspect per-realization output:
  [Working with Results](results.md).
