# Realization Filters

A realization filter selects, for each evaluation batch, which realizations
contribute to the combined function or gradient value. Filters enable
risk-aware optimization (for example, focusing on the worst-performing
realizations) and common variance-reduction techniques.

`ropt` ships a **CVaR filter** in the `ropt.realization_filter.default`
module, which selects the realizations contributing to the
Conditional-Value-at-Risk tail. It can be configured for objectives, for
constraints, or both.

## Selecting a filter

Filters are listed at the top level of the configuration and referenced from the
objectives or nonlinear constraints that use them, so different objectives can be
filtered differently. This one optimizes the conditional expectation of the worst
30% of 10 realizations:

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

Omit both fields to leave every realization at its configured weight. At each
evaluation a filter returns per-realization weights that replace
`realizations.weights` for that evaluation. See [Sharing optimizer components by
key](configuration.md#sharing-optimizer-components-by-key) for the indexing
pattern, and [`realization_filters`](configuration_sections.md#realization-filters)
for the fields of a filter configuration.

See [`CVaRObjectiveOptions`][ropt.realization_filter.default.CVaRObjectiveOptions]
for the parameters. The corresponding constraint variant is
[`CVaRConstraintOptions`][ropt.realization_filter.default.CVaRConstraintOptions].

## Interaction with `evaluation_policy`

Filters that disable some realizations only deliver savings on the gradient
side when the optimizer requests gradients separately from functions. Set
`gradient.evaluation_policy = "separate"` (see
[Stochastic Gradients](gradients.md)) to maximize that benefit.

## Writing a custom filter

Custom filters are plugins implementing the
[`RealizationFilter`][ropt.realization_filter.RealizationFilter] base class,
whose abstract methods define what a filter must provide. A filter defined where
an entry point cannot reach it — in a script or a notebook — is added with
[`register_plugin`][ropt.plugins.register_plugin], after which it is
selected by its `"plugin/method"` string exactly like an installed one. An
instance can also be passed directly in the `realization_filters` field of
[`EnOptContext`][ropt.context.EnOptContext], which needs no registration.

The runnable script is
[examples/simple/realization_filter.py](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/realization_filter.py),
which implements a filter that puts all weight on the median realization,
registers it, and selects it from the configuration as `"custom/median"`.

## How CVaR filters work

The `cvar-objective` method:

1. Computes a weighted sum of the objective values specified by the `sort`
   indices for each realization (using the objective weights from the
   configuration). If a single objective index is given, no weighting is
   applied. The objective [scales](configuration_sections.md#objective-scales) are
   applied first, and objectives marked in
   [`maximize`](configuration_sections.md#objective-direction) have their sign flipped,
   per objective, so that the sum ranks realizations the way the optimizer
   would.
2. Conceptually sorts realizations by that value, worst first. Since step 1
   flipped the sign of any maximized objective, the worst outcomes are the
   highest values.
3. Takes the `percentile` fraction of realizations from the start of that
   order.
4. Gives each selected realization the same weight, one divided by the number
   of valid realizations. If `percentile` does not cover a whole number of
   realizations, the realization at the boundary receives the leftover
   fraction instead. All other realizations receive zero.
5. Failed realizations (NaN values) are excluded.

The `cvar-constraint` variant applies the same steps to a single constraint
function, named by `sort`, ranking realizations by their violation of that
constraint, largest first. The violation is the distance to the violated side,
`maximum(lower - c, c - upper)`, where `c` is the constraint value and `lower`
and `upper` are the bounds it was configured with. It is positive when the
constraint is violated and negative by the amount of slack otherwise, and one
expression covers the three kinds of constraint:

- Bounded from above: the largest values are the most violated.
- Bounded from below: the smallest values are the most violated.
- An equality: the values furthest from the bound, in either direction.

!!! note
    Realizations reach a filter with their objectives and constraints exactly
    as the evaluator returned them: neither scaled nor flipped for direction,
    since both belong to the aggregate and these are per-realization values. A
    filter that ranks by what the optimizer minimizes applies them itself, as
    the CVaR filter does. A constraint has no direction to apply, and the
    constraint bounds reach the filter in the same unscaled domain as the
    values, so the two can be subtracted directly. A constraint scale, being
    positive, leaves the order within one constraint unchanged.

!!! note "Weight normalization"
    The optimizer normalizes all filter-produced weights to sum to one before
    use, so any non-negative values are permissible.
