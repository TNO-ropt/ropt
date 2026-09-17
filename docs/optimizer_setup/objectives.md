# Objectives

An optimizer needs a single number to minimize. When a problem has more than one
quantity worth improving — a cost and a risk, a yield and an emission — `ropt`
forms that number for you, as a weighted sum of the objectives you declare.

This page covers how many objectives there are, how they are weighted, which
direction each is optimized in, and how scaling interacts with the weights. For
the field-by-field reference, see
[`objectives`](configuration_sections.md#objectives).

## Declaring more than one objective

The `weights` field declares how many objectives there are: one entry each.
Nothing else in the configuration changes.

```python
CONFIG = {
    "variables": {"variable_count": 5, "perturbation_magnitudes": 1e-6},
    "objectives": {"weights": [0.75, 0.25]},
}
```

An evaluation function with several objectives returns a sequence instead of a
single number, in the order the weights declare them:

```python
def objectives(variables, context):
    return [cost(variables), risk(variables)]
```

The order is positional — there are no names — and if the problem also has
nonlinear constraints, their values follow the objectives in the same sequence.
See [Constraints](constraints.md) and
[Running Optimizations](../running/running.md#the-evaluation-function).

Two properties of `weights` matter here:

- **Weights are normalized to sum to one.** `[1, 1]` and `[0.5, 0.5]` describe
  the same problem, so only the ratio matters.
- **A zero weight disables an objective.** It is still evaluated and still
  reported, but it does not influence the search. Weights may not be negative.

[examples/simple/export.py](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/export.py)
runs a two-objective problem, and gives the objectives names so that the
exported table is readable.

## Maximizing instead of minimizing

`ropt` minimizes. To maximize an objective, mark it in `maximize`, which takes
one boolean per objective:

```python
"objectives": {"weights": [0.6, 0.4], "maximize": [False, True]}
```

Do not try to maximize by making a weight negative; weights must be
non-negative, and `maximize` is the supported way to express a direction. See
[Choosing the direction of an
objective](configuration_sections.md#objective-direction).

The sign is flipped **after** the realizations have been combined, never on the
per-realization values. That ordering matters as soon as a
[function estimator](function_estimators.md) produces something other than an
average. Negating the inputs of a standard deviation leaves it unchanged, so
flipping first would silently minimize the dispersion that was declared for
maximization; flipping the combined value is correct whatever produced it.

## Weights and scales decide the trade-off together

Objectives reach the optimizer divided by their `scales`, so what is actually
minimized is

$$ \sum_j \frac{w_j}{s_j} f_j $$

The effective weight of an objective is therefore $w_j / s_j$, not $w_j$. With a
single objective this does not affect the solution: dividing by a positive
constant leaves the optimum where it is. With several it does, because changing
one scale changes the balance between them and moves the optimum with it.

The consequence is that two objectives differing by orders of magnitude are not
balanced by their weights alone — the larger one dominates whatever the weights
say, because it contributes far more to the sum:

```python
"objectives": {"weights": [0.5, 0.5], "scales": [1.0, 1e6]}
```

Setting `auto_scale` instead estimates a *single* factor from the first batch,
which brings the weighted total near one while preserving the objectives'
relative magnitudes — so it does not rebalance them, and does not move the
optimum.

!!! warning "A rescaled run reports a different objective"

    `auto_scale` divides the objective the optimizer reports, so switching it on
    lowers `target_objective` without improving the solution. Two runs that
    differ in this setting can only be compared through `functions.objectives`
    or their variables. The runnable demonstration is
    [examples/simple/scaling.py](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/scaling.py).

An `offsets` entry is subtracted before the division. It cannot move the optimum
or change the gradient, but it changes the magnitude the optimizer tests against
its tolerances, which is what makes it useful for an objective that is large and
varies little. See [Scaling
objectives](configuration_sections.md#objective-scales) and
[Offsetting objectives](configuration_sections.md#objective-offsets) for the
full field descriptions.

## See also

- What a run reports in each domain, and which fields are comparable between
  runs: [Working with Results](../running/results.md#scaling-of-results).
