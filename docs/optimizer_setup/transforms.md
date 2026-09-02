# Transforms

A transform converts a quantity between the **user domain** (the values you
specify in the configuration and read back from results) and the **optimizer
domain** (the values actually presented to the optimization algorithm).

Three independent transform types exist:

| Transform                                                                 | Applied to                                       |
| ------------------------------------------------------------------------- | ------------------------------------------------ |
| [`VariableTransform`][ropt.transforms.VariableTransform]                  | Optimization variables (and their bounds).       |
| [`ObjectiveTransform`][ropt.transforms.ObjectiveTransform]                | Objective function values.                       |
| [`NonlinearConstraintTransform`][ropt.transforms.NonlinearConstraintTransform] | Nonlinear constraint values.                |

## Why use them

- **Scaling** — bring variables of different physical magnitudes onto a common
  numerical scale so the optimizer treats them evenly.
- **Reparametrization** — optimize in a more convenient space (log-space,
  normalized $[0,1]$ space).
- **Stability** — apply monotone transforms to objectives that span many
  orders of magnitude.

## How they plug in

Transform instances are stored in top-level tuples on the
[`EnOptContext`][ropt.context.EnOptContext]:

- `variable_transforms`
- `objective_transforms`
- `nonlinear_constraint_transforms`

Each tuple is an ordered **chain** applied to *all* variables, objectives, or
constraints. Going to the optimizer domain the chain runs in order; coming back
it runs in reverse:

```text
user domain  --[ A ]-->--[ B ]-->--[ C ]-->  optimizer domain
user domain  <--[ A ]--<--[ B ]--<--[ C ]--  optimizer domain
```

```python
"variable_transforms": [
    {"method": "default/scaler", "options": {"offsets": [10.0, 0.0, 0.0]}},
    {"method": "default/scaler", "options": {"scales": [1.0, 1e3, 1e-3]}},
],
"variables": {"variable_count": 3},
```

Here every variable is first shifted, then scaled. Reading results back, the
values are unscaled first and unshifted second.

Because a chain applies to everything, order matters whenever the transforms do
not commute — as in the example above, where shifting and scaling give
different answers depending on which comes first.

Variables held fixed through the `mask` field of `variables` are never passed
to the optimizer, and the chain leaves them unchanged.

### Restricting a transform to a subset

Each transform config takes an optional boolean `mask` selecting the elements
it applies to. It defaults to `None`, meaning the whole set:

```python
"objective_transforms": [
    {
        "method": "default/scaler",
        "options": {"scales": [1e6, 1.0]},
        "mask": [True, False],   # only the first objective is scaled
    },
],
```

Unlike the chain itself, masks of different transforms need not be disjoint: an
element is acted on by every chain member whose mask selects it, in order. A
mask is therefore a filter, not a partition.

Masks are available on all three transform configs, and the transform decides
what to do with them — the core does not apply them for you. The built-in
scalers implement a mask by neutralizing their scale and offset on the excluded
positions, which leaves those values untouched in both directions.

For variable transforms the mask is combined with the free-variable mask by a
logical *and*, so a variable is transformed only when it is both free and
selected.

## Default transforms

The `ropt.transforms.default` package provides built-in linear scaling
transforms, exposed as the `default/scaler` method.

### DefaultVariableTransform

[`DefaultVariableTransform`][ropt.transforms.default.DefaultVariableTransform]
applies a per-variable linear scale and offset:

$$x_{\text{opt}} = \frac{x_{\text{user}} - \text{offset}}{\text{scale}}, \qquad
  x_{\text{user}} = x_{\text{opt}} \cdot \text{scale} + \text{offset}$$

Configuration options:

- **`scales`** — array of per-variable scaling factors (default: no scaling).
- **`offsets`** — array of per-variable offsets (default: no offset).

When both are provided they are broadcasted to the same length.

This transform also handles:

- **Perturbation magnitudes** — divided by `scale` so perturbations remain
  proportional in optimizer space.
- **Variable bound differences** — multiplied by `scale` when reporting
  constraint violations back in user units.
- **Linear constraints** — the coefficient matrix $\mathbf{A}$ and RHS bounds
  $\mathbf{b}$ are adjusted to account for the scale and offset (see the
  [API reference][ropt.transforms.default.DefaultVariableTransform.linear_constraints_to_optimizer]
  for the full derivation). The resulting equations are further normalized by
  dividing each row by its maximum absolute coefficient.

### DefaultObjectiveTransform

[`DefaultObjectiveTransform`][ropt.transforms.default.DefaultObjectiveTransform]
divides objective values by `scales` when going to the optimizer domain and
multiplies when returning:

$$f_{\text{opt}} = f_{\text{user}} / \text{scale}, \qquad
  f_{\text{user}} = f_{\text{opt}} \cdot \text{scale}$$

Configuration options:

- **`scales`** — array of per-objective scaling factors.

The `update(scales)` method allows changing scales mid-run (for example, for
adaptive normalization when initial magnitudes are unknown).

### DefaultNonlinearConstraintTransform

[`DefaultNonlinearConstraintTransform`][ropt.transforms.default.DefaultNonlinearConstraintTransform]
divides constraint values *and* their RHS bounds by `scales`:

$$c_{\text{opt}} = c_{\text{user}} / \text{scale}, \qquad
  b_{\text{opt}} = b_{\text{user}} / \text{scale}$$

Configuration options:

- **`scales`** — array of per-constraint scaling factors.

Like the objective transform, it supports `update(scales)` for mid-run
changes. Constraint-violation differences are multiplied by `scales` when
converting back to user domain.

## Effects on bounds and constraints

During context initialization, `ropt` automatically applies the variable
transform chain to:

- Variable **lower and upper bounds** (via `to_optimizer`).
- **Perturbation magnitudes** (via `magnitudes_to_optimizer`).
- **Linear constraint** coefficients and RHS bounds (via
  `linear_constraints_to_optimizer`).

This happens once at startup, so you always specify bounds, magnitudes, and
linear constraints in user-domain terms. See
[`LinearConstraintsConfig`][ropt.config.LinearConstraintsConfig] for the
underlying math.

## Round-tripping results

During optimization, objective/constraint values are computed in the user
domain by the evaluator, then transformed to the optimizer domain for the
algorithm. Results objects therefore live in the **optimizer domain** by
default.

To obtain user-domain results, call
[`transform_from_optimizer`][ropt.results.Results.transform_from_optimizer] on
a `FunctionResults` or `GradientResults` object. This returns a new results
object with variables, objectives, and constraints mapped back to user-domain
values (including bound/constraint violation differences).

Higher-level helpers handle this automatically:

- The [simple API](../running/running.md) returns user-domain results.
- Event handlers may or may not perform the conversion automatically. For
  example, [`ResultsHandler`][ropt.components.event_handlers.ResultsHandler] accepts
  a `domain` argument (`"user"` or `"optimizer"`) to control which domain its
  stored result lives in.

## Writing a custom transform

Custom transforms are plugins implementing one of the three base classes —
[`VariableTransform`][ropt.transforms.VariableTransform],
[`ObjectiveTransform`][ropt.transforms.ObjectiveTransform], or
[`NonlinearConstraintTransform`][ropt.transforms.NonlinearConstraintTransform]
— whose docstrings document the lifecycle and the methods to implement.
`VariableTransform` is the most involved: variables also interact with bounds
and linear constraints, and it is the only one that receives the free-variable
mask, through `set_free_mask`.

A transform only needs to describe its own step; `ropt` takes care of running
the chain in the right order in each direction. `to_optimizer` and
`from_optimizer` must be inverses, otherwise results cannot be mapped back.

## Where to next

- [Realization Filters](realization_filters.md) — selecting realizations per
  objective or constraint.
- [Configuration](configuration.md) — broadcasting and index-sharing rules in
  context.
