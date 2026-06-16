# Realization Filters

A realization filter selects, for each evaluation batch, which realizations
contribute to the combined function or gradient value. Filters enable
risk-aware optimization (e.g., focus on the worst-performing realizations) and
common variance-reduction tricks.

`ropt` ships with two default filters in the
`ropt.realization_filter.default` module:

- A **sorting filter** that keeps the worst (or best) `N` realizations.
- A **CVaR filter** that selects realizations contributing to the
  Conditional-Value-at-Risk tail.

Each filter can be configured for objectives, for constraints, or both.

## How filters fit in

1. You add filter configurations to the top-level `realization_filters` list.
2. You point objectives (or constraints) at a filter by its index in
   `ObjectiveFunctionsConfig.realization_filters` /
   `NonlinearConstraintsConfig.realization_filters`.
3. At each evaluation, the filter is consulted to compute per-realization
   weights that override the static `realizations.weights`.

See [Configuration](configuration.md) for the index-sharing pattern.

## Worst-`N` example (sorting filter)

Optimize the average of the 3 worst realizations out of 10:

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
            "method": "default/sort-objective",
            "options": {"sort": "descending", "first": 0, "last": 2},
        },
    ],
    "gradient": {"number_of_perturbations": 5},
}
```

Options are validated against
[`SortObjectiveOptions`][ropt.realization_filter.default.SortObjectiveOptions].

### How sorting filters work

The `sort-objective` method:

1. Computes a weighted sum of the objective values specified by the `sort`
   indices for each realization (using the objective weights from the
   configuration). If a single objective index is given, no weighting is
   applied.
2. Sorts realizations by that value (ascending).
3. Selects realizations whose rank falls in the inclusive range
   \[`first`, `last`\].
4. Retains the original realization weights for selected realizations; all
   others receive zero. Failed realizations (NaN values) are given the lowest
   rank and excluded before selection.

The `sort-constraint` variant
([`SortConstraintOptions`][ropt.realization_filter.default.SortConstraintOptions])
works identically but sorts on a single constraint function value.

## CVaR example

Optimize the conditional expectation of the worst 30% of realizations:

```python
"realization_filters": [
    {
        "method": "default/cvar-objective",
        "options": {"percentile": 0.3},
    },
],
"objectives": {"weights": [1.0], "realization_filters": [0]},
```

See [`CVaRObjectiveOptions`][ropt.realization_filter.default.CVaRObjectiveOptions]
for the parameters. The corresponding constraint variant is
[`CVaRConstraintOptions`][ropt.realization_filter.default.CVaRConstraintOptions].

### How CVaR filters work

The `cvar-objective` method:

1. Computes a weighted sum of objectives (same as the sorting filter).
2. Conceptually sorts realizations by that value (ascending, assuming
   minimization).
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

!!! note "Weight normalization"
    The optimizer normalizes all filter-produced weights to sum to one before
    use, so any non-negative values are permissible.

## Interaction with `evaluation_policy`

Filters that disable some realizations only deliver savings on the gradient
side when the optimizer requests gradients separately from functions. Set
`gradient.evaluation_policy = "separate"` (see
[Stochastic Gradients](gradients.md)) to maximize that benefit.

## Writing a custom filter

Custom filters are implemented as plugins. See the
[`RealizationFilter`][ropt.realization_filter.RealizationFilter] base class.

A filter subclass must implement three methods:

1. `__init__(filter_config)` — store the configuration; keep setup lightweight.
2. `init(context)` — called once with the
   [`EnOptContext`][ropt.context.EnOptContext] after all configuration is
   finalized. Perform validation and precomputation here.
3. `get_realization_weights(objectives, constraints)` — called at each
   evaluation. Return a 1-D array of non-negative weights (one per
   realization).

Registering a filter with the plugin system is only required when the filter
should be selectable via
[`RealizationFilterConfig`][ropt.config.RealizationFilterConfig]. Otherwise,
filter instances can be passed directly in the `realization_filters` field of
[`EnOptContext`][ropt.context.EnOptContext].

## Where to next

- Combine filters with transforms:
  [Transforms](transforms.md).
- Inspect per-realization output:
  [Working with Results](results.md).
