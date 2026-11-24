# Realization Aggregators

A realization aggregator combines per-realization objective or constraint
values (and their gradients) into the representative values used by the
optimizer. In ensemble-based optimization each realization produces its own
function and gradient values; the aggregator combines them according to a chosen
strategy.

`ropt` ships with a default aggregator in the
`ropt.realization_aggregator.default` module that provides two methods:

- **`mean`** (or **`default`**): weighted average of realization values and
  gradients — the standard approach for expected-value optimization.
- **`stddev`**: weighted standard deviation of realization values with
  chain-rule gradients — useful when the optimization target is variability
  rather than the mean.

## Selecting a realization aggregator

Realization aggregators are configured in the top-level `realization_aggregator` field and
referenced from the objectives or nonlinear constraints that use them, so
different objectives can be aggregated differently. The default method is a
weighted average:

```python
CONFIG = {
    "variables": {"variable_count": 3, "perturbation_magnitudes": 1e-5},
    "realizations": {"weights": [1.0] * 10},
    "objectives": {
        "weights": [1.0],
        "realization_aggregators": [0],   # objective uses aggregator 0
    },
    "realization_aggregator": {"method": "default/mean"},
}
```

Because `mean` is the default, both fields can be omitted entirely when
weighted-average aggregation is all you need. During optimization the aggregator
receives the per-realization function and gradient arrays together with the
current weights, and returns an aggregated value. See [Sharing optimizer
components by index](configuration.md#sharing-optimizer-components-by-index) for
the indexing pattern, and
[`realization_aggregator`](configuration_sections.md#realization-aggregator)
for the fields of an aggregator configuration.

## Standard-deviation aggregation

To optimize for low variability instead of low mean, use `stddev`:

```python
"realization_aggregator": {"method": "default/stddev"},
"objectives": {"weights": [1.0], "realization_aggregators": [0]},
```

Note:

- At least two realizations with non-zero weight are required.
- The `stddev` method is incompatible with `gradient.merge_realizations = True`;
  per-realization gradients must be available.
- A standard deviation is always positive. To *maximize* variability, set
  [`maximize`](objectives.md#maximizing-instead-of-minimizing) on the objective;
  the sign is flipped after the standard deviation has been computed, which is
  the only point at which flipping it has any effect.

## Writing a custom aggregator

Custom aggregators are plugins implementing the
[`RealizationAggregator`][ropt.realization_aggregator.RealizationAggregator] base class,
whose docstring documents the methods to implement, including how
`merge_realizations` changes what `aggregate_gradient` receives. Registering
an aggregator with the plugin system is only required when it should be
selectable via [`RealizationAggregatorConfig`][ropt.config.RealizationAggregatorConfig];
otherwise, an instance can be passed directly in the `realization_aggregator`
section of the configuration.
For unmerged gradients, `aggregate_gradient` receives an array with shape
`(n_realizations, n_variables)`; with merged realizations, it receives
`(n_variables,)`.

The runnable script is
[examples/realization_aggregator.py](https://github.com/TNO-ropt/ropt/blob/main/examples/realization_aggregator.py),
which implements a weighted geometric mean, registers it, and selects it from
the configuration as `"custom/geometric"`.
