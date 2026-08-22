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

Each variable, objective, or constraint is assigned to a transform by its
**index** in the corresponding `transforms` array inside `VariablesConfig`,
`ObjectiveFunctionsConfig`, or `NonlinearConstraintsConfig`. An index of `-1`
(the default) means no transform applies.

```python
"variable_transforms": [
    {"method": "default/scaler", "options": {"scales": [1.0, 1e3, 1e-3]}},
],
"variables": {
    "variable_count": 3,
    "transforms": [0, 0, 0],   # all three variables use transform 0
},
```

Multiple transforms may exist, each handling a different subset of variables or
objectives/constraints.

At context initialization, each transform's `init(mask)` method is called
with a boolean mask combining:

- For variables: which variables are free (`variables.mask`) **and** assigned
  to this transform (`variables.transforms == idx`).
- For objectives/constraints: which objectives or constraints are assigned to
  this transform.

The transform must treat unmasked elements as identity (pass through
unchanged).

See [Configuration](configuration.md) for the broadcasting and index-sharing
rules.

## Default transforms

The `ropt.plugins.transforms.default` package provides built-in linear scaling
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

During context initialization, `ropt` automatically applies variable transforms
to:

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
`VariableTransform` is the most involved, since variables also interact with
bounds and linear constraints.

## Where to next

- [Realization Filters](realization_filters.md) — also configured via
  index-sharing.
- [Configuration](configuration.md) — broadcasting and index-sharing rules in
  context.
