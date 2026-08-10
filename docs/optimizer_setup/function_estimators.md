# Function Estimators

A function estimator aggregates per-realization objective or constraint values
(and their gradients) into the single representative values used by the
optimizer. In ensemble-based optimization each realization produces its own
function and gradient values; the estimator combines them according to a chosen
strategy.

`ropt` ships with a default estimator in the
`ropt.function_estimator.default` module that provides two methods:

- **`mean`** (or **`default`**): weighted average of realization values and
  gradients — the standard approach for expected-value optimization.
- **`stddev`**: weighted standard deviation of realization values with
  chain-rule gradients — useful when the optimization target is variability
  rather than the mean.

## How estimators fit in

1. You add estimator configurations to the top-level `function_estimators`
   list in the context.
2. You point objectives (or constraints) at an estimator by its index in
   `ObjectiveFunctionsConfig.function_estimators` /
   `NonlinearConstraintsConfig.function_estimators`.
3. During optimization, the estimator is called with per-realization function
   and gradient arrays plus the current weights, and returns a single
   aggregated value.

See [Configuration](configuration.md) for the index-sharing pattern.

## Mean estimator (default)

The default method computes a simple weighted average:

```python
CONFIG = {
    "variables": {"variable_count": 3, "perturbation_magnitudes": 1e-5},
    "realizations": {"weights": [1.0] * 10},
    "objectives": {
        "weights": [1.0],
        "function_estimators": [0],   # objective uses estimator 0
    },
    "function_estimators": [
        {"method": "default/mean"},   # index 0
    ],
}
```

Because `mean` is the default, you can omit the `function_estimators` list
entirely when weighted-average aggregation is all you need.

## Standard-deviation estimator

To optimize for low variability instead of low mean, use `stddev`:

```python
"function_estimators": [
    {"method": "default/stddev"},
],
"objectives": {"weights": [1.0], "function_estimators": [0]},
```

Note:

- At least two realizations with non-zero weight are required.
- The `stddev` method is incompatible with `gradient.merge_realizations = True`;
  per-realization gradients must be available.

## Writing a custom estimator

Custom estimators are implemented as plugins. See the
[`FunctionEstimator`][ropt.function_estimator.FunctionEstimator] base class for
the interface you need to implement.

A subclass must implement four methods:

1. `__init__(estimator_config)` — store the configuration; keep setup
   lightweight.
2. `init(context)` — called once with the
   [`EnOptContext`][ropt.context.EnOptContext] after configuration is
   finalized. Validate settings (e.g., check compatibility with
   `merge_realizations`) and pre-compute any state here.
3. `calculate_function(functions, weights)` — aggregate per-realization
   function values into a single representative value.
    - `functions`: shape `(n_realizations,)`.
    - `weights`: shape `(n_realizations,)`.
    - Returns: a scalar or 1-D array.
4. `calculate_gradient(functions, gradient, weights)` — aggregate
   per-realization gradients into a single gradient vector.
    - `functions`: shape `(n_realizations,)` — needed for chain-rule estimators.
    - `gradient`: shape `(n_realizations, n_variables)` (or `(n_variables,)` when
      `merge_realizations=True`).
    - `weights`: shape `(n_realizations,)`.
    - Returns: 1-D array of shape `(n_variables,)`.

### The `merge_realizations` setting

The [`GradientConfig.merge_realizations`][ropt.config.GradientConfig] flag
controls how gradients are presented to the estimator:

- `False` (default): `ropt` estimates a separate gradient per realization. The
  estimator combines them using weights.
- `True`: `ropt` estimates a single merged gradient from all perturbations
  collectively. Suitable only for averaging-type estimators. If your estimator
  is incompatible (e.g., `stddev` needs per-realization gradients), raise
  `ValueError` from `init`.

Registering the estimator with the plugin system is only required when it
should be selectable via
[`FunctionEstimatorConfig`][ropt.config.FunctionEstimatorConfig]. Otherwise,
instances can be passed directly in the `function_estimators` field of
[`EnOptContext`][ropt.context.EnOptContext].

## Where to next

- Filter realizations before aggregation:
  [Realization Filters](realization_filters.md).
- Transform aggregated values:
  [Transforms](transforms.md).
