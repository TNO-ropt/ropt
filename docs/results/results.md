# Working with Results

`ropt` exposes the full intermediate and final state of an optimization through
[`Results`][ropt.results.Results] objects. This page describes the result
classes and how to inspect them; see
[Running Optimizations](../running/running.md) for how results are produced and
delivered to your code.

!!! note "One kind of result object"

    Evaluation data uses the same [`Results`][ropt.results.Results] classes
    throughout `ropt`. Handlers and result callbacks receive these objects,
    and the return values of [`optimize`](../running/running.md) and
    [`evaluate`](../running/running.md#evaluating-without-optimizing) expose
    them through their `results` field. Each object describes one evaluated
    variable vector. Its fields have the same meaning regardless of how you
    obtain it.

## The result hierarchy

Result handlers receive [`EnOptEvent`][ropt.events.EnOptEvent] objects:
notifications that identify an optimization or evaluation event and carry
the configuration and any associated results. When an evaluation finishes,
the event's `results` field contains the function or gradient results
produced. A result callback receives the individual result objects rather
than the event itself. See [Writing your own handler](handlers.md#writing-your-own-handler)
for the event types and how to handle them.

Each [`Results`][ropt.results.Results] object represents the outcome of the
calculation for a **single variable vector** — that is, the objective and
gradient values computed at one point in variable space. However, the optimizer
may request evaluations at multiple variable vectors in a single batch (for
example, multiple perturbations or multiple candidates in a gradient-free
method). In that case, the event payload contains a *sequence* of `Results`
objects, one per variable vector evaluated in that batch.

Two concrete subclasses exist:

- [`FunctionResults`][ropt.results.FunctionResults] — objective and constraint
  values for a batch.
- [`GradientResults`][ropt.results.GradientResults] — gradient estimates for
  the objective and constraints.

### Fields of function and gradient results { #what-each-field-holds }

The fields below describe the data stored in `FunctionResults` and
`GradientResults`. Some fields hold arrays directly; others group related
data in [`ResultField`][ropt.results.ResultField] objects. For example, a
`FunctionResults` object stores per-realization objective values in
`evaluations.objectives` and their aggregates in `functions.objectives`.

#### `FunctionResults` fields

- **`variables`** — the variable vector that was evaluated.
- **`target_objective`** — the single weighted scalar the optimizer minimizes
  (0-D array), or `None` if no aggregate could be formed. Always in the domain
  the optimizer works in; see [Scaling of results](#scaling-of-results).
- **`evaluations`** ([`FunctionEvaluations`][ropt.results.FunctionEvaluations])
  — the raw per-realization values returned by the evaluator:
    - `objectives`: objective values per realization.
    - `constraints`: constraint values per realization (only present when
      nonlinear constraints are configured).
    - `metadata`: optional dict of per-realization metadata arrays.
- **`functions`** ([`Functions`][ropt.results.Functions]) — aggregated values
  derived from the per-realization evaluations (or `None` if too few
  realizations succeeded):
    - `objectives`: individual objective values.
    - `constraints`: individual constraint values (if configured).
- **`realizations`** ([`Realizations`][ropt.results.Realizations]) — ensemble
  metadata:
    - `evaluated_realizations`: boolean array indicating which realizations were
      evaluated.
    - `objective_weights`: per-realization objective weights, or `None` when no
      [realization filter](../optimizer_setup/realization_filters.md) is
      configured or every realization failed. A filter may change them from one
      batch to the next.
    - `constraint_weights`: per-realization constraint weights, or `None` unless
      nonlinear constraints and a realization filter are both configured and at
      least one realization succeeded.
- **`constraint_info`** ([`ConstraintInfo`][ropt.results.ConstraintInfo]) —
  constraint bound information. Present when bounds or constraints are defined.
  Contains two kinds of data for each constraint type (bound, linear, and
  nonlinear):

    - **Differences**: the signed distance between the current value and each
      bound. For lower bounds, a negative difference means the value is below
      the bound (violated). For upper bounds, a positive difference means the
      value is above the bound (violated).
    - **Violations**: the absolute magnitude of any bound exceedance, zero
      when the constraint is satisfied, or `NaN` when the value is `NaN`. For
      example, if a constraint requires $g(\mathbf{x}) \leq 0$ and the actual
      value is $0.5$, the violation is $0.5$.

    See the [`ConstraintInfo`][ropt.results.ConstraintInfo] reference for
    the full list of fields.

#### `GradientResults` fields

- **`variables`** — the unperturbed variable vector.
- **`perturbed_variables`** — perturbed variable values.
- **`target_gradient`** — the gradient the optimizer descends, or `None` if
  estimation failed. Always in the domain the optimizer works in.
- **`evaluations`** ([`GradientEvaluations`][ropt.results.GradientEvaluations])
  — the raw per-perturbation values returned by the evaluator:
    - `perturbed_objectives`: objective values for each perturbation.
    - `perturbed_constraints`: constraint values for each perturbation (if
      configured).
    - `metadata`: optional dict of per-realization/perturbation metadata
      arrays.
- **`gradients`** ([`Gradients`][ropt.results.Gradients]) — aggregated gradient
  values (or `None` if estimation failed):
    - `objectives`: per-objective gradients.
    - `constraints`: per-constraint gradients (if configured).
- **`realizations`** ([`Realizations`][ropt.results.Realizations]) — same
  structure as for `FunctionResults` (see above).

The shape and meaning of each array dimension are listed under
[Axes and dimensionality](#axes-and-dimensionality). All values are NumPy
arrays.

### Common attributes on all results

Every [`Results`][ropt.results.Results] object carries:

- **`batch_id`**: an integer identifying the evaluation batch
  (potentially generated by the evaluator).
- **`metadata`**: a dictionary of additional information generated during
  optimization. Not interpreted by `ropt` — useful for reporting and analysis.
- **`names`**: a mapping from dimension names to label tuples. Keys are
  [`AxisName`][ropt.enums.AxisName] values, or the name of a metadata key that
  adds a [user-defined dimension](#user-defined-axes). These labels are used
  to produce multi-index DataFrames when exporting (see
  [Exporting to pandas](#exporting-to-pandas)).

## Accessing result data

Common access patterns:

```python
result.variables                   # variable vector evaluated
result.target_objective            # weighted scalar objective
result.functions.objectives        # per-objective values
result.functions.constraints       # per-constraint values
```

If `functions` is `None`, the result represents a request that produced no
valid values (for example, fewer realizations succeeded than
[`realization_min_success`](../optimizer_setup/configuration_sections.md#realizations)
requires). `target_objective` is `None` exactly then, so a single guard covers
both:

```python
if result.functions is not None:
    print(result.target_objective)
```

With `realization_min_success` set to zero, `functions` is not `None` even when
all realizations failed: its values and `target_objective` are `NaN`.

## Axes and dimensionality

Much of the data within result objects is multi-dimensional. For example, the
`objectives` field within
[`FunctionEvaluations`][ropt.results.FunctionEvaluations] is a 2-D array where
each row is a realization and each column is an objective.

To support exporting and reporting, `ropt` records what each dimension
represents as axis metadata on each field. The
[`ResultField`][ropt.results.ResultField] base class provides a
[`get_axes`][ropt.results.AxisMetadata.get_axes] class method for retrieving
this metadata:

```python
from ropt.results import FunctionEvaluations

FunctionEvaluations.get_axes("objectives")
# (<AxisName.REALIZATION: 'realization'>, <AxisName.OBJECTIVE: 'objective'>)
```

The [`AxisName`][ropt.enums.AxisName] enumeration defines:

| Axis name              | Meaning
| ---------------------- |---------
| `VARIABLE`             | Index corresponds to the variable number as defined in [`VariablesConfig`][ropt.config.VariablesConfig].
| `OBJECTIVE`            | Index corresponds to the objective number (position in the `weights` array of [`ObjectiveFunctionsConfig`][ropt.config.ObjectiveFunctionsConfig]).
| `NONLINEAR_CONSTRAINT` | Index corresponds to the nonlinear constraint number as defined in [`NonlinearConstraintsConfig`][ropt.config.NonlinearConstraintsConfig].
| `LINEAR_CONSTRAINT`    | Index corresponds to the linear constraint number as defined in [`LinearConstraintsConfig`][ropt.config.LinearConstraintsConfig].
| `REALIZATION`          | Index corresponds to the realization number in the ensemble. Present whenever results involve multiple realizations.
| `PERTURBATION`         | Index corresponds to a perturbation used for gradient estimation. Present in [`GradientEvaluations`][ropt.results.GradientEvaluations] where objectives and constraints are reported for each perturbed variable set.

The number and order of dimensions are fixed for each field. Throughout, $n_v$ is
the number of variables, $n_o$ of objectives, $n_c$ of nonlinear constraints,
$n_l$ of linear constraints, $n_r$ of realizations and $n_p$ of perturbations.

**Function results**

=== "Variables"

    `variables`: the vector of variable values at which the functions were
    evaluated.

    - Shape $(n_v,)$
    - Axes: [`VARIABLE`][ropt.enums.AxisName.VARIABLE]

=== "Target objective"

    `target_objective`: the single weighted value the optimizer minimizes.

    - Shape $()$ — a 0-D array with no dimensions.

=== "Objectives"

    `evaluations.objectives`: the objective values for each realization, one
    row per realization and one column per objective.

    - Shape $(n_r, n_o)$
    - Axes: [`REALIZATION`][ropt.enums.AxisName.REALIZATION],
      [`OBJECTIVE`][ropt.enums.AxisName.OBJECTIVE]

    `functions.objectives`: the same values aggregated over the ensemble.

    - Shape $(n_o,)$
    - Axes: [`OBJECTIVE`][ropt.enums.AxisName.OBJECTIVE]

=== "Constraints"

    `evaluations.constraints`: the nonlinear constraint values for each
    realization. Only present when nonlinear constraints are configured.

    - Shape $(n_r, n_c)$
    - Axes: [`REALIZATION`][ropt.enums.AxisName.REALIZATION],
      [`NONLINEAR_CONSTRAINT`][ropt.enums.AxisName.NONLINEAR_CONSTRAINT]

    `functions.constraints`: the same values aggregated over the ensemble.

    - Shape $(n_c,)$
    - Axes: [`NONLINEAR_CONSTRAINT`][ropt.enums.AxisName.NONLINEAR_CONSTRAINT]

=== "Realizations"

    `realizations.evaluated_realizations`: which realizations were evaluated.

    - Shape $(n_r,)$
    - Axes: [`REALIZATION`][ropt.enums.AxisName.REALIZATION]

    `realizations.objective_weights` and `realizations.constraint_weights`:
    per-realization weights, `None` unless a realization filter is configured.

    - Shape $(n_o, n_r)$ and $(n_c, n_r)$
    - Axes: [`OBJECTIVE`][ropt.enums.AxisName.OBJECTIVE] or
      [`NONLINEAR_CONSTRAINT`][ropt.enums.AxisName.NONLINEAR_CONSTRAINT], then
      [`REALIZATION`][ropt.enums.AxisName.REALIZATION]

=== "Constraint info"

    `constraint_info.*`: the differences and violations for each constraint
    type. Each `bound_`, `linear_` and `nonlinear_` field has one axis.

    - Shapes $(n_v,)$, $(n_l,)$ and $(n_c,)$
    - Axes: [`VARIABLE`][ropt.enums.AxisName.VARIABLE],
      [`LINEAR_CONSTRAINT`][ropt.enums.AxisName.LINEAR_CONSTRAINT] or
      [`NONLINEAR_CONSTRAINT`][ropt.enums.AxisName.NONLINEAR_CONSTRAINT]

=== "Metadata"

    `evaluations.metadata`: optional per-realization metadata from the
    evaluator. Each entry is a one-dimensional array of any type `numpy`
    supports, including objects, so each key may have its own dtype.

    - Shape $(n_r,)$
    - Axes: [`REALIZATION`][ropt.enums.AxisName.REALIZATION]

    An entry whose values are arrays rather than scalars has shape
    $(n_r, n_k)$ and a second dimension for its elements, named after its key.
    See [User-defined axes](#user-defined-axes).

**Gradient results**

=== "Variables"

    `variables`: the vector of unperturbed variable values.

    - Shape $(n_v,)$
    - Axes: [`VARIABLE`][ropt.enums.AxisName.VARIABLE]

=== "Target gradient"

    `target_gradient`: the gradient the optimizer descends.

    - Shape $(n_v,)$
    - Axes: [`VARIABLE`][ropt.enums.AxisName.VARIABLE]

=== "Perturbed variables"

    `perturbed_variables`: the perturbed variable values for each realization
    and perturbation.

    - Shape $(n_r, n_p, n_v)$
    - Axes: [`REALIZATION`][ropt.enums.AxisName.REALIZATION],
      [`PERTURBATION`][ropt.enums.AxisName.PERTURBATION],
      [`VARIABLE`][ropt.enums.AxisName.VARIABLE]

=== "Perturbed objectives"

    `evaluations.perturbed_objectives`: the objective values for each
    realization and perturbation.

    - Shape $(n_r, n_p, n_o)$
    - Axes: [`REALIZATION`][ropt.enums.AxisName.REALIZATION],
      [`PERTURBATION`][ropt.enums.AxisName.PERTURBATION],
      [`OBJECTIVE`][ropt.enums.AxisName.OBJECTIVE]

    `gradients.objectives`: the gradient estimated from them, per objective.

    - Shape $(n_o, n_v)$
    - Axes: [`OBJECTIVE`][ropt.enums.AxisName.OBJECTIVE],
      [`VARIABLE`][ropt.enums.AxisName.VARIABLE]

=== "Perturbed constraints"

    `evaluations.perturbed_constraints`: the nonlinear constraint values for
    each realization and perturbation. Only present when nonlinear constraints
    are configured.

    - Shape $(n_r, n_p, n_c)$
    - Axes: [`REALIZATION`][ropt.enums.AxisName.REALIZATION],
      [`PERTURBATION`][ropt.enums.AxisName.PERTURBATION],
      [`NONLINEAR_CONSTRAINT`][ropt.enums.AxisName.NONLINEAR_CONSTRAINT]

    `gradients.constraints`: the gradient estimated from them, per constraint.

    - Shape $(n_c, n_v)$
    - Axes: [`NONLINEAR_CONSTRAINT`][ropt.enums.AxisName.NONLINEAR_CONSTRAINT],
      [`VARIABLE`][ropt.enums.AxisName.VARIABLE]

=== "Realizations"

    `realizations.*`: the same fields and shapes as for function results.

=== "Metadata"

    `evaluations.metadata`: optional metadata per realization and
    perturbation, each entry a two-dimensional array of any `numpy` type.

    - Shape $(n_r, n_p)$
    - Axes: [`REALIZATION`][ropt.enums.AxisName.REALIZATION],
      [`PERTURBATION`][ropt.enums.AxisName.PERTURBATION]

    An entry of arrays rather than scalars has shape $(n_r, n_p, n_k)$ and a
    third dimension for the metadata elements, named after its key.

**Scaled results**

[`ScaledFunctionResults`][ropt.results.ScaledFunctionResults] and
[`ScaledGradientResults`][ropt.results.ScaledGradientResults] repeat a subset of
the fields above in the optimizer's domain, with identical shapes and
dimensions:
`variables`, `functions` and `constraint_info` for the first, and `variables`,
`perturbed_variables` and `gradients` for the second. See
[Scaling of results](#scaling-of-results).

### User-defined axes

Per-realization metadata is the one place where you can add a dimension of your
own. If the evaluation function returns an **array** instead of a scalar for a
metadata key, the result has one extra dimension for the array's elements,
named after the key. The API calls this a user-defined axis:

```python
return EvaluationFunctionResult(objectives=value, metadata={"residual": residual})
```

With three realizations and a residual of length three,
`result.evaluations.metadata["residual"]` has shape $(3, 3)$. Each row
corresponds to a realization, and each column to an element of `residual`.
Label those elements by adding an entry to `names` under the same key:

```python
"names": {"residual": ("x", "y", "z")},
```

Every realization must return the same number of entries for a key, and a key
may not be named after an [`AxisName`][ropt.enums.AxisName] value or
`batch_id`; both raise a `ValueError`. If the metadata elements are not
labelled, their labels default to integer indices.

!!! note
    Array dimensions are fixed: even with a single objective, result arrays
    still include an `OBJECTIVE` dimension of length one.

## Scaling of results

Optimization internally works with scaled values: variables are scaled and
shifted by their
[`scales` and `offsets`](../optimizer_setup/configuration_sections.md#variable-scales), objective and
nonlinear constraint *aggregates* have their
[offsets](../optimizer_setup/configuration_sections.md#objective-offsets) subtracted and are divided by
their [scales](../optimizer_setup/configuration_sections.md#objective-scales), and objectives marked
[`maximize`](../optimizer_setup/configuration_sections.md#objective-direction) are negated once they have
been combined across realizations.

Every result carries both domains at once, and one rule connects them:

!!! note
    `scaled.X` is the optimizer's version of `X`, at the same path. Fields
    without a scaled counterpart have only one domain.

So `result.variables` is the variable vector as configured and
`result.scaled.variables` is the same vector as the optimizer proposed it;
`result.functions.objectives` and `result.scaled.functions.objectives` are the
same pair for the aggregates.

Two groups of fields have a single domain:

- The per-realization values in `evaluations` are reported exactly as the
  evaluator returned them. Scales apply to the quantities the optimizer
  consumes, and the optimizer never sees a single realization, so there is
  nothing to undo for those fields.
- `target_objective` and `target_gradient` exist only in the domain the
  optimizer works in. Each is a weighted total over objectives that may differ
  in both scale and direction, so there is no single factor to undo. The
  gradient is differentiated with respect to the *scaled* variables.

Because the direction is undone when reporting, a combined objective agrees in
sign with the per-realization values it summarizes, whether it is an average or
a measure of dispersion.

The runnable script is
[examples/scaling.py](https://github.com/TNO-ropt/ropt/blob/main/examples/scaling.py),
which runs the same problem unscaled, with `auto_scale` on the objective, and
with a variable scale, printing both domains each time. It shows that
`auto_scale` divides the reported objective by a large factor while leaving the
solution where it was, so only `functions.objectives` and `variables` can be
compared between optimizations.

## Metadata

Results carry two independent kinds of metadata, neither interpreted by `ropt`:

- **Result metadata** — the `metadata` dict on every
  [`Results`][ropt.results.Results] object. Pass defaults as `metadata=` to
  [`optimize`][ropt.optimize], [`evaluate`][ropt.evaluate], or
  [`evaluate_batch`][ropt.evaluate_batch], for example `{"run_id": 7}` to
  identify a run. An
  [event handler for `START_EVALUATION`](../running/running.md#passing-values-to-evaluations-from-event-handlers)
  can add or override values for one batch. The evaluation function receives
  the effective values as `context.metadata`, and each result records an
  independent copy. Later updates do not change earlier results.
- **Per-realization metadata** — the `metadata` dict on the `evaluations` field,
  with one array entry per realization. It is produced by the objective when it
  returns an
  [`EvaluationFunctionResult`][ropt.EvaluationFunctionResult]
  with a `metadata` field.

Result metadata is passed to `optimize` or `evaluate` and read back from
`metadata`:

```python
result = optimize(config, x0, objective, metadata={"run_id": 7})
result.results.metadata            # {'run_id': 7}
```

Per-realization metadata is returned by the objective and read back from the
`evaluations` field, with one entry per realization:

```python
def objective(variables, context):
    ...
    return EvaluationFunctionResult(objectives=value, metadata={"shift": shift})


result.results.evaluations.metadata   # {'shift': array([...])}
```

The full runnable script is
[examples/metadata.py](https://github.com/TNO-ropt/ropt/blob/main/examples/metadata.py).

## Exporting results

Results can be exported to pandas or Polars DataFrames. The two formats use
the same field selection and unstacking options, but represent row labels
differently. Install the corresponding `pandas` or `polars` optional extra;
see [Installation](../getting_started/installation.md).

Use [`to_pandas`][ropt.results.Results.to_pandas] or
[`to_polars`][ropt.results.Results.to_polars] for a single result, and
[`results_to_pandas`][ropt.results.results_to_pandas] or
[`results_to_polars`][ropt.results.results_to_polars] for a sequence of results.

The runnable script is
[examples/export.py](https://github.com/TNO-ropt/ropt/blob/main/examples/export.py).
It uses Polars by default and pandas with `--pandas`, so only one of the two
needs to be installed.

!!! note

    These functions turn results into DataFrames. The
    [`DataFrameHandler`](handlers.md#dataframehandler) uses them to build and
    update such tables for you as an optimization runs; see
    [Result Handlers](handlers.md).

### Selecting fields and labels

Row-label dimensions use their [`AxisName`][ropt.enums.AxisName] names,
such as `"variable"`, `"realization"` and `"objective"`. The export also
includes `batch_id` to distinguish results from different batches.

The row labels and the unstacked column labels come from the
[`names`](../optimizer_setup/configuration_sections.md#names) mapping in the
configuration. If a dimension is not named, its labels fall back to 0-based
integer indices. For example, exporting the objectives of a single result
**without** any `names` gives plain numbers for both the realizations and
objectives:

```python
pandas_frame = result.to_pandas(["evaluations.objectives"])
polars_frame = result.to_polars(["evaluations.objectives"])
```

For compactness, the shared examples below show the pandas representation.
Polars contains the same selected values; the
[format differences](#exporting-to-polars) are described after the shared options.

```
                               evaluations.objectives
batch_id realization objective
1        0           0                           2.10
                     1                           0.94
         1           0                           2.35
                     1                           1.02
```

Adding a `names` entry replaces those numbers with meaningful labels:

```python
CONFIG = {
    ...
    "names": {
        "variable": ("x0", "x1", "x2"),
        "realization": ("r0", "r1"),
        "objective": ("val", "cost"),
    },
}
```

The examples below use these labels.

Both single-result methods export a list of fields, each named by a dotted
path from the result:

```python
pandas_frame = result.to_pandas(["variables", "evaluations.objectives"])
polars_frame = result.to_polars(["variables", "evaluations.objectives"])
```

A path may name a field of the result itself (`"variables"`,
`"target_objective"`), a field of one of its sub-objects
(`"functions.objectives"`, `"scaled.variables"`), or an entry of a dict-valued
field (`"metadata.run.id"`). Each path becomes a column of that name. Paths
whose value is `None`, and missing dict keys, are skipped; descending into a
field that is `None` is an error.

!!! note "Fields are an ordered list"

    Every export function takes its fields as a **sequence**, and the columns
    follow the order you give. Passing a set raises a `TypeError`, because a
    set has no order to follow, and naming the same path twice raises a
    `ValueError`, because one column cannot appear twice.

### Moving dimensions into columns

By default, each dimension of an exported field becomes a row label.
For example, `objectives` in
[`FunctionEvaluations`][ropt.results.FunctionEvaluations] has dimensions
identified by `REALIZATION` and `OBJECTIVE`, so exporting it keeps both in the
row labels — now with the configured names:

```python
df = result.to_pandas(["evaluations.objectives"])
```

```
                               evaluations.objectives
batch_id realization objective
1        r0          val                         2.10
                     cost                        0.94
         r1          val                         2.35
                     cost                        1.02
```

Passing `unstack` moves selected dimensions from row labels into columns.
Here the `OBJECTIVE` dimension is unstacked:

```python
from ropt.enums import AxisName

df = result.to_pandas(
    ["evaluations.objectives"],
    unstack=[AxisName.OBJECTIVE],
)
# Use result.to_polars(...) with the same arguments for Polars.
```

```
                     (evaluations.objectives, val)  (evaluations.objectives, cost)
batch_id realization
1        r0                                   2.10                            0.94
         r1                                   2.35                            1.02
```

Each new column identifies the full field path and the labels of its
unstacked dimensions. Pandas uses tuple column labels; Polars joins the field
path and dimension labels into a string, as described below. Unstacking every
dimension of every selected field leaves one row per result, which makes
results from different batches comparable row by row.

### Aggregating multiple results

`results_to_pandas` and `results_to_polars` convert a *sequence* of results
into a single DataFrame. Neither takes an `unstack` argument, because both
always pivot the same way:

!!! note
    Every dimension becomes columns **except those for realizations and
    perturbations**, which stay as row labels next to `batch_id`.

A field with neither of those dimensions therefore gives exactly one row per
result:

```python
from ropt import results_to_pandas, results_to_polars

df = results_to_pandas(
    all_results,
    fields=["variables"],
    result_type="functions",
)
# Use results_to_polars(...) with the same arguments for Polars.
```

```
          (variables, x0)  (variables, x1)  (variables, x2)
batch_id
1                    0.30             0.42            -0.11
2                    0.55             0.48             0.02
3                    0.61             0.50             0.10
```

Here each row is one result identified by its `batch_id`; selecting a field
that keeps the `realization` dimension adds a row label and one row per
realization. Field names use dot
notation for nested sub-fields (for example, `variables`, `target_objective`),
and the fields appear in the order you list them. The `result_type`
argument selects which results to process: `"functions"` for
[`FunctionResults`][ropt.results.FunctionResults] only, `"gradients"` for
[`GradientResults`][ropt.results.GradientResults] only.

### Metadata columns

Both kinds of [metadata](#metadata) are reached by the same dotted paths, since
a path may end in one or more dict keys.

The **per-realization metadata** attached by the evaluator lives on the
evaluations, so it has one entry for each realization. For example, if the
objective attached a per-realization `shift`:

```python
df = result.to_pandas(["evaluations.metadata.shift"])
```

```
                     evaluations.metadata.shift
batch_id realization
1        r0                                 0.9
         r1                                 1.1
```

If the value a realization returns is an array rather than a scalar, the
metadata also has a dimension for its elements, named after the key. This
[user-defined dimension](#user-defined-axes) is exported like the built-in
dimensions. Exporting a single result keeps it as a row label, here for a
two-element `residual` labelled `"x"`/`"y"`:

```python
df = result.to_pandas(["evaluations.metadata.residual"])
```

```
                               evaluations.metadata.residual
batch_id realization residual
1        r0          x                                  0.12
                     y                                 -0.04
         r1          x                                  0.31
                     y                                  0.08
```

Passing `unstack=["residual"]` moves it into columns, just like a built-in
dimension. In an aggregated frame the rule above applies, so the metadata
elements are always columns while `realization` stays a row label:

```python
df = results_to_pandas(
    all_results,
    fields=["evaluations.metadata.residual"],
    result_type="functions",
)
```

```
                      (evaluations.metadata.residual, x)  (evaluations.metadata.residual, y)
batch_id realization
1        r0                                         0.12                               -0.04
         r1                                         0.31                                0.08
2        r0                                         0.09                               -0.02
         r1                                         0.22                                0.05
```

The **result metadata** records the run defaults and any batch-local overrides
on each result. A scalar entry, such as a tag identifying the optimization,
gives one value per result. A one-dimensional array adds a dimension named
after its key, just as array-valued per-realization metadata does. Single-result
exports keep that dimension as a row label unless it is unstacked.
When exporting several results with `results_to_pandas` or `results_to_polars`,
each array position becomes a separate column. For example,
`metadata.scores = [10, 20]` produces two columns containing `10` and `20`
on the row for that result.

Result metadata can contain dictionaries nested to any depth. Their entries
are selected using dotted paths in both the single-result and sequence export
functions:

```python
df = results_to_pandas(
    all_results,
    fields=["metadata.run_id", "target_objective"],
    result_type="functions",
)
```

```
          metadata.run_id  target_objective
batch_id
1                       0              1.83
2                       1              0.42
3                       2              0.11
```

### pandas and Polars layout differences

#### Exporting to pandas

Pandas stores row labels in an index, with one level per retained dimension.
Unstacked column labels are tuples of the field path and dimension labels,
such as `("evaluations.objectives", "val")`. Each additional unstacked
dimension adds an element to the tuple.

#### Exporting to polars

Polars has no index and its column names must be strings, which leads to the two
differences you need to know about. The frames below are shown without the
borders polars draws around them in a terminal.

**Index levels become ordinary columns.** What pandas puts in the index, polars
puts in leading columns of the frame:

```python
df = result.to_polars(["evaluations.objectives"])
```

```
batch_id  realization  objective  evaluations.objectives
1         r0           val        2.10
1         r0           cost       0.94
1         r1           val        2.35
1         r1           cost       1.02
```

**Tuple column labels become joined strings.** Where pandas produces the column
`("evaluations.objectives", "val")`, polars produces
`"evaluations.objectives,val"`. The separator is configurable with the `sep`
argument, which defaults to `","`:

```python
from ropt.enums import AxisName

df = result.to_polars(
    ["evaluations.objectives"],
    unstack=[AxisName.OBJECTIVE],
)
```

```
batch_id  realization  evaluations.objectives,val  evaluations.objectives,cost
1         r0           2.10                        0.94
1         r1           2.35                        1.02
```

!!! note

    Because polars keeps the keys as real columns, it can join fields that vary
    at different granularities — for example a per-batch gradient with
    per-perturbation evaluations — by repeating the coarser values across the
    finer rows. Pandas cannot align such fields and returns them as disjoint
    blocks of rows padded with missing values instead. Only polars produces a
    single table that mixes granularities.
