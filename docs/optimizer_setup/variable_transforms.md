# Variable Transforms

A [`VariableTransform`][ropt.transforms.VariableTransform] converts variables
between the **user domain** (the values you specify in the configuration and
read back from results) and the **optimizer domain** (the values actually
presented to the optimization algorithm).

## Why use them

- **Scaling** — bring variables of different physical magnitudes onto a common
  numerical scale so the optimizer treats them evenly.
- **Reparametrization** — optimize in a more convenient space (log-space,
  normalized $[0,1]$ space).

## How they plug in

Transform instances are stored in the `variable_transforms` tuple on the
[`EnOptContext`][ropt.context.EnOptContext]. The tuple is an ordered **chain**
applied to *all* variables. Going to the optimizer domain the chain runs in
order; coming back it runs in reverse:

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

A transform config takes an optional boolean `mask` selecting the variables it
applies to. It defaults to `None`, meaning all of them:

```python
"variable_transforms": [
    {
        "method": "default/scaler",
        "options": {"scales": [1e6, 1.0, 1.0]},
        "mask": [True, False, False],   # only the first variable is scaled
    },
],
```

Unlike the chain itself, masks of different transforms need not be disjoint: an
element is acted on by every chain member whose mask selects it, in order. A
mask is therefore a filter, not a partition.

The transform decides what to do with a mask — the core does not apply it for
you. The built-in scaler implements a mask by neutralizing its scale and offset
on the excluded positions, which leaves those values untouched in both
directions.

The mask is combined with the free-variable mask by a logical *and*, so a
variable is transformed only when it is both free and selected.

## The default transform

The `ropt.transforms.default` package provides a built-in linear scaling
transform, exposed as the `default/scaler` method.

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

## Effects on bounds and linear constraints

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

Results objects live in the **optimizer domain** by default, so the variables
they carry are the transformed ones.

To obtain user-domain results, call
[`transform_from_optimizer`][ropt.results.Results.transform_from_optimizer] on
a `FunctionResults` or `GradientResults` object. It returns a new results
object with the chain run in reverse, including the bound and linear-constraint
differences. See [Results](results.md) for what each field holds in each
domain.

Higher-level helpers handle this automatically:

- The [simple API](../running/running.md) returns user-domain results.
- Event handlers may or may not perform the conversion automatically. For
  example, [`ResultsHandler`][ropt.components.event_handlers.ResultsHandler] accepts
  a `domain` argument (`"user"` or `"optimizer"`) to control which domain its
  stored result lives in.

## Writing a custom transform

A custom transform is a plugin implementing
[`VariableTransform`][ropt.transforms.VariableTransform], whose docstring
documents the lifecycle and the methods to implement. Variables interact with
bounds and linear constraints, and the transform receives the free-variable
mask through `set_free_mask`.

A transform only needs to describe its own step; `ropt` takes care of running
the chain in the right order in each direction. `to_optimizer` and
`from_optimizer` must be inverses, otherwise results cannot be mapped back.

## Where to next

- [Results](results.md) — which domain each result field lives in.
- [Configuration](configuration.md) — broadcasting rules and the remaining
  context fields, including scaling.
