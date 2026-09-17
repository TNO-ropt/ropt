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

## Selecting an estimator

Estimators are listed at the top level of the configuration and referenced from
the objectives or nonlinear constraints that use them, so different objectives
can be aggregated differently. The default method is a weighted average:

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

Because `mean` is the default, both fields can be omitted entirely when
weighted-average aggregation is all you need. During optimization the estimator
receives the per-realization function and gradient arrays together with the
current weights, and returns a single aggregated value. See [Sharing optimizer
components by key](configuration.md#sharing-optimizer-components-by-key) for the
indexing pattern, and
[`function_estimators`](configuration_sections.md#function-estimators)
for the fields of an estimator configuration.

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
- A standard deviation is always positive. To *maximize* variability, set
  [`maximize`](objectives.md#maximizing-instead-of-minimizing) on the objective;
  the sign is flipped after the standard deviation has been computed, which is
  the only point at which flipping it has any effect.

## Writing a custom estimator

Custom estimators are plugins implementing the
[`FunctionEstimator`][ropt.function_estimator.FunctionEstimator] base class,
whose docstring documents the methods to implement, including how
`merge_realizations` changes what `calculate_gradient` receives. Registering
an estimator with the plugin system is only required when it should be
selectable via [`FunctionEstimatorConfig`][ropt.config.FunctionEstimatorConfig];
otherwise, an instance can be passed directly in the `function_estimators`
field of [`EnOptContext`][ropt.context.EnOptContext].

The runnable script is
[examples/simple/function_estimator.py](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/function_estimator.py),
which implements a weighted geometric mean, registers it, and selects it from
the configuration as `"custom/geometric"`.
