# Working with Results

`ropt` exposes the full intermediate and final state of an optimization through
[`Results`][ropt.results.Results] objects. This page describes the result
classes and how to inspect them; see [Running Optimizations](running.md) and
[Optimization Workflows](../advanced/workflows.md) for how results are produced and
delivered to your code.

!!! note "Two layers of result object"

    [`Results`][ropt.results.Results] is the **fundamental** record: one object
    per variable vector evaluated, carrying every field described below. Result
    handlers receive these.

    [`EvaluateResult`][ropt.simple.EvaluateResult] and
    [`OptimizeResult`][ropt.simple.OptimizeResult], which
    [`optimize`](running.md) returns and a `report` callback is given, are a
    **convenience** layer over one of them: a few named attributes such as
    `variables`, `target_objective` and `exit_code`, for the common case where
    that is all you need. The fundamental object is still there, on
    `result.results`, whenever it is not.

## The result hierarchy

During optimization, function and gradient evaluations generate data that is
reported via [`EnOptEvent`][ropt.events.EnOptEvent] objects passed to callbacks.

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

Each carries nested [`ResultField`][ropt.results.ResultField] objects:

| Result             | Fields                                                                                          |
| ------------------ | ----------------------------------------------------------------------------------------------- |
| `FunctionResults`  | `variables`, `target_objective`, `evaluations` ([`FunctionEvaluations`][ropt.results.FunctionEvaluations]), `functions` ([`Functions`][ropt.results.Functions]), `realizations` ([`Realizations`][ropt.results.Realizations]), `constraint_info` ([`ConstraintInfo`][ropt.results.ConstraintInfo]), `scaled` ([`ScaledFunctionResults`][ropt.results.ScaledFunctionResults]). |
| `GradientResults`  | `variables`, `perturbed_variables`, `target_gradient`, `evaluations` ([`GradientEvaluations`][ropt.results.GradientEvaluations]), `gradients` ([`Gradients`][ropt.results.Gradients]), `scaled` ([`ScaledGradientResults`][ropt.results.ScaledGradientResults]). |

### What each field holds

#### `FunctionResults` fields

- **`variables`** — the variable vector that was evaluated, shape $(n_v,)$.
- **`target_objective`** — the single weighted scalar the optimizer minimizes
  (0-D array), or `None` if no aggregate could be formed. Always in the domain
  the optimizer works in; see [Scaling of results](#scaling-of-results).
- **`evaluations`** ([`FunctionEvaluations`][ropt.results.FunctionEvaluations])
  — the raw per-realization values returned by the evaluator:
    - `objectives`: objective values per realization, shape $(n_r, n_o)$.
    - `constraints`: constraint values per realization, shape $(n_r, n_c)$
      (only present when nonlinear constraints are configured).
    - `metadata`: optional dict of per-realization metadata arrays, each of
      shape $(n_r,)$.
- **`functions`** ([`Functions`][ropt.results.Functions]) — aggregated values
  derived from the per-realization evaluations (or `None` if all realizations
  failed):
    - `objectives`: individual objective values, shape $(n_o,)$.
    - `constraints`: individual constraint values, shape $(n_c,)$ (if
      configured).
- **`realizations`** ([`Realizations`][ropt.results.Realizations]) — ensemble
  metadata:
    - `evaluated_realizations`: boolean array indicating which realizations were
      evaluated, shape $(n_r,)$.
    - `objective_weights`: per-realization objective weights, shape
      $(n_o, n_r)$, or `None` when no
      [realization filter](../optimizer_setup/realization_filters.md) is
      configured. A filter may change them from one batch to the next.
    - `constraint_weights`: per-realization constraint weights, shape
      $(n_c, n_r)$, or `None` unless nonlinear constraints and a realization
      filter are both configured.
- **`constraint_info`** ([`ConstraintInfo`][ropt.results.ConstraintInfo]) —
  constraint bound information. Present when bounds or constraints are defined.
  Contains two kinds of data for each constraint type (bound, linear, and
  nonlinear):

    - **Differences**: the signed distance between the current value and each
      bound. For lower bounds, a negative difference means the value is below
      the bound (violated). For upper bounds, a positive difference means the
      value is above the bound (violated).
    - **Violations**: the absolute magnitude of any bound exceedance, or zero
      when the constraint is satisfied. For example, if a constraint requires
      $g(\mathbf{x}) \leq 0$ and the actual value is $0.5$, the violation is
      $0.5$.

    See the [`ConstraintInfo`][ropt.results.ConstraintInfo] reference for
    the full list of fields.

#### `GradientResults` fields

- **`variables`** — the unperturbed variable vector, shape $(n_v,)$.
- **`perturbed_variables`** — perturbed variable values, shape
  $(n_r, n_p, n_v)$.
- **`target_gradient`** — the gradient the optimizer descends, shape $(n_v,)$,
  or `None` if estimation failed. Always in the domain the optimizer works in.
- **`evaluations`** ([`GradientEvaluations`][ropt.results.GradientEvaluations])
  — the raw per-perturbation values returned by the evaluator:
    - `perturbed_objectives`: objective values for each perturbation, shape
      $(n_r, n_p, n_o)$.
    - `perturbed_constraints`: constraint values for each perturbation, shape
      $(n_r, n_p, n_c)$ (if configured).
    - `metadata`: optional dict of per-realization/perturbation metadata
      arrays, each of shape $(n_r, n_p)$.
- **`gradients`** ([`Gradients`][ropt.results.Gradients]) — aggregated gradient
  values (or `None` if estimation failed):
    - `objectives`: per-objective gradients, shape $(n_o, n_v)$.
    - `constraints`: per-constraint gradients, shape $(n_c, n_v)$ (if
      configured).
- **`realizations`** ([`Realizations`][ropt.results.Realizations]) — same
  structure as for `FunctionResults` (see above).

In the shapes above: $n_v$ = number of variables, $n_o$ = number of objectives,
$n_c$ = number of nonlinear constraints, $n_r$ = number of realizations,
$n_p$ = number of perturbations. All values are NumPy arrays.

### Common attributes on all results

Every [`Results`][ropt.results.Results] object carries:

- **`batch_id`**: an integer identifying the evaluation batch
  (potentially generated by the evaluator).
- **`metadata`**: a dictionary of additional information generated during
  optimization. Not interpreted by `ropt` — useful for reporting and analysis.
- **`names`**: a mapping from axis name to label tuples. Keys are
  [`AxisName`][ropt.enums.AxisName] values, or the name of a metadata key that
  defines a [user axis](#user-defined-axes). Used to produce labelled
  multi-index DataFrames when exporting (see
  [Exporting to pandas](#exporting-to-pandas)).

## Accessing result data

Common access patterns:

```python
result.variables                   # variable vector evaluated
result.target_objective            # weighted scalar objective
result.functions.objectives        # per-objective values (after weighting)
result.functions.constraints       # per-constraint values
```

If `functions` is `None`, the result represents a request that produced no
valid values (for example, all realizations failed). `target_objective` is
`None` exactly then, so a single guard covers both:

```python
if result.functions is not None:
    print(result.target_objective)
```

## Axes and dimensionality

Much of the data within result objects is multi-dimensional. For example, the
`objectives` field within
[`FunctionEvaluations`][ropt.results.FunctionEvaluations] is a 2-D array where
each row is a realization and each column is an objective.

To simplify exporting and reporting, the identity of each dimension is stored as
axis metadata on each field. The [`ResultField`][ropt.results.ResultField] base
class provides a [`get_axes`][ropt.results.AxisMetadata.get_axes] class method
for retrieving this metadata:

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

The dimensionality and order of axes for each field are fixed — they are listed
in the "Result descriptions" section of each class in the
[reference](../reference/results.md).

### User-defined axes

Per-realization metadata is the one place where you can add an axis of your own.
If the evaluation function returns an **array** instead of a scalar for a
metadata key, that key gains one extra axis, named after the key itself:

```python
return EvaluationFunctionResult(objectives=value, metadata={"residual": residual})
```

With three realizations and a residual of length three,
`result.evaluations.metadata["residual"]` has shape $(3, 3)$ and axes
`("realization", "residual")`. Label the new axis by adding an entry to `names`
under the same key:

```python
"names": {"residual": ("x", "y", "z")},
```

Every realization must return the same number of entries for a key, and a key
may not be named after an [`AxisName`][ropt.enums.AxisName] value or
`batch_id`; both raise a `ValueError`. Unlabelled user axes fall back to integer
indices, as builtin axes do.

!!! note
    Dimensionality is fixed: even with a single objective, result arrays still
    include an `OBJECTIVE` axis of length one.

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
  gradient is differentiated with respect to the *scaled* variables. If you need
  either in configured terms, combine the objectives yourself using
  [`get_objective_scales`][ropt.context.EnOptContext.get_objective_scales] and
  the directions on `objectives.maximize`.

Because the direction is undone when reporting, a combined objective agrees in
sign with the per-realization values it summarizes, whether it is an average or
a measure of dispersion.

The runnable script is
[examples/simple/scaling.py](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/scaling.py),
which runs the same problem unscaled, with `auto_scale` on the objective, and
with a variable scale, printing both domains each time. It shows that
`auto_scale` divides the reported objective by a large factor while leaving the
solution where it was, so only `functions.objectives` and `variables` can be
compared between runs.

## Metadata

Results carry two independent kinds of metadata, neither interpreted by `ropt`:

- **Result metadata** — the `metadata` dict on every
  [`Results`][ropt.results.Results] object, identical for every result of a run.
  It is set once when the run starts: pass a `metadata` dict to the simple-API
  [`optimize`][ropt.simple.optimize] /
  [`optimize_many`][ropt.simple.optimize_many] /
  [`evaluate`][ropt.simple.evaluate] functions (or to the low-level compute
  step). Use it to tag or identify a run, for example `{"run_id": 7}`.
- **Per-realization metadata** — the `metadata` dict on the `evaluations` field,
  with one array entry per realization. It is produced by the objective when it
  returns an
  [`EvaluationFunctionResult`][ropt.components.evaluators.EvaluationFunctionResult]
  with a `metadata` field.

Result metadata is passed to the run and read back from `metadata`:

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
[examples/simple/metadata.py](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/metadata.py).

## Exporting to pandas

`ropt` can export results to `pandas` DataFrames for analysis and reporting.
This requires the `pandas` optional extra (see [Installation](../getting_started/installation.md)).

!!! note

    These functions turn results into DataFrames. The
    [`DataFrameHandler`](handlers.md#dataframehandler) uses them to build and
    update such tables for you as an optimization runs; see
    [Result Handlers](handlers.md).

The row index and the unstacked column labels come from the
[`names`](../optimizer_setup/configuration_sections.md#names) mapping in the configuration. If an axis is
not named, its labels fall back to 0-based integer indices. For example,
exporting the objectives of a single result **without** any `names` gives plain
numbers for both the realization and the objective axes:

```python
df = result.to_pandas(["evaluations.objectives"])
```

```
                               evaluations.objectives
batch_id realization objective
1        0           0                           2.10
                     1                           0.94
         1           0                           2.35
                     1                           1.02
```

Adding a `names` entry replaces those numbers with meaningful labels. The
examples below assume the realizations are named `"r0"`/`"r1"` and the objectives
`"val"`/`"cost"`.

### Exporting selected fields

The [`to_pandas`][ropt.results.Results.to_pandas] method on an individual
result exports a list of fields, each named by a dotted path from the result:

```python
df = result.to_pandas(["variables", "evaluations.objectives"])
```

The runnable script for this section is
[examples/simple/export.py](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/export.py),
which exports one result stacked, the same result unstacked, and then every
result of the run in one frame. It uses polars by default and pandas with
`--pandas`, so only one of the two needs to be installed.

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

By default, every axis of the exported fields becomes a level in a
multi-index. For example, `objectives` in
[`FunctionEvaluations`][ropt.results.FunctionEvaluations] has the axes
`REALIZATION` and `OBJECTIVE`, so exporting it keeps both in the index — now
with the configured names:

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

Passing `unstack` pivots selected axes out of the index and into columns. Here
the `OBJECTIVE` axis is unstacked:

```python
from ropt.enums import AxisName

df = result.to_pandas(
    ["evaluations.objectives"],
    unstack=[AxisName.OBJECTIVE],
)
```

```
                     (evaluations.objectives, val)  (evaluations.objectives, cost)
batch_id realization
1        r0                                   2.10                            0.94
         r1                                   2.35                            1.02
```

The unstacked axis is flattened into the column labels, so each new column is a
tuple of the full field path and the axis label — here
`("evaluations.objectives", "val")` and `("evaluations.objectives", "cost")`.
Unstacking more axes adds more elements to these tuples. Unstacking every axis
of every selected field leaves one row per result, which is what makes results
from different batches comparable row by row.

### Aggregating multiple results

[`results_to_pandas`][ropt.results.results_to_pandas] builds on `to_pandas` to
convert a *sequence* of results into a single DataFrame. It takes no `unstack`
argument, because an aggregated frame always pivots the same way:

!!! note
    Every axis becomes columns **except `realization` and `perturbation`**,
    which stay as row labels next to `batch_id`.

A field with neither of those axes therefore gives exactly one row per result:

```python
from ropt.results import results_to_pandas

df = results_to_pandas(
    all_results,
    fields=["variables"],
    result_type="functions",
)
```

```
          (variables, x0)  (variables, x1)  (variables, x2)
batch_id
1                    0.30             0.42            -0.11
2                    0.55             0.48             0.02
3                    0.61             0.50             0.10
```

Each column is a `(field, label)` pair, and here each row is one result
identified by its `batch_id`; selecting a field that keeps the `realization`
axis adds a level to the index and one row per realization. Field names use dot
notation for nested sub-fields (for example, `variables`, `target_objective`),
and the fields appear in the order you list them. The `result_type`
argument selects which results to process: `"functions"` for
[`FunctionResults`][ropt.results.FunctionResults] only, `"gradients"` for
[`GradientResults`][ropt.results.GradientResults] only.

### Metadata columns

Both kinds of [metadata](#metadata) are reached by the same dotted paths, since
a path may end in one or more dict keys.

The **per-realization metadata** attached by the evaluator lives on the
evaluations, so it keeps the `realization` axis. For example, if the objective
attached a per-realization `shift`:

```python
df = result.to_pandas(["evaluations.metadata.shift"])
```

```
                     evaluations.metadata.shift
batch_id realization
1        r0                                 0.9
         r1                                 1.1
```

If the value a realization returns is an array rather than a scalar, the key
also spans a [user axis](#user-defined-axes) named after the key, which is
treated like a builtin one. Exporting a single result keeps it in the index,
here for a two-element `residual` labelled `"x"`/`"y"`:

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

Passing `unstack=["residual"]` pivots it into columns, exactly as for a builtin
axis. In an aggregated frame the rule above applies, so the axis is always
unstacked while `realization` stays a row label:

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

The run-level **result metadata** sits directly on the result, so it has no
axes and gives one value per result — handy for pulling in a run tag. It may be
nested to any depth, and is reachable from `to_pandas` and `results_to_pandas`
alike:

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

### Labels and the index

Every axis of an exported field becomes an index level, named after its
[`AxisName`][ropt.enums.AxisName] value (for example `"variable"`, `"realization"`,
`"objective"`), and `batch_id` is always prepended so results from different
batches stay distinct. The label on each level — and on each unstacked column —
comes from the [`names`](../optimizer_setup/configuration_sections.md#names) mapping in the configuration, a
dict from axis name to a tuple of labels:

```python
CONFIG = {
    ...
    "names": {
        "variable": ("x0", "x1", "x2"),
        "objective": ("val", "cost"),
    },
}
```

An axis without a `names` entry falls back to 0-based integer indices, as in the
first example of this section.

## Exporting to polars

Every pandas export has a `polars` counterpart:
[`to_polars`][ropt.results.Results.to_polars] and
[`results_to_polars`][ropt.results.results_to_polars]. They accept the same
arguments and select and unstack exactly the same fields, so everything in the
previous section carries over. This requires the `polars` optional extra (see
[Installation](../getting_started/installation.md)).

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

Aggregating a sequence of results works the same way:

```python
from ropt.results import results_to_polars

df = results_to_polars(
    all_results,
    fields=["variables"],
    result_type="functions",
)
```

```
batch_id  variables,x0  variables,x1  variables,x2
1         0.30          0.42          -0.11
2         0.55          0.48          0.02
3         0.61          0.50          0.10
```

Metadata behaves exactly as described [above](#metadata-columns), and both kinds
are reachable from either function:

```python
df = results_to_polars(
    all_results,
    fields=["metadata.run_id", "target_objective"],
    result_type="functions",
)
```

```
batch_id  metadata.run_id  target_objective
1         0                1.83
2         1                0.42
3         2                0.11
```

!!! note

    Because polars keeps the keys as real columns, it can join fields that vary
    at different granularities — for example a per-batch gradient with
    per-perturbation evaluations — by repeating the coarser values across the
    finer rows. Pandas cannot align such fields and returns them as disjoint
    blocks of rows padded with missing values instead, so prefer polars when a
    single table has to mix granularities.
