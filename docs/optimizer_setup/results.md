# Working with Results

`ropt` exposes the full intermediate and final state of an optimization through
[`Results`][ropt.results.Results] objects. This page describes the result
classes and how to inspect them; see [Running Optimizations](../running/running.md) and
[Optimization Workflows](../workflows/workflows.md) for how results are produced and
delivered to your code.

## The result hierarchy

During optimization, function and gradient evaluations generate data that is
reported via [`EnOptEvent`][ropt.events.EnOptEvent] objects passed to callbacks.

Each [`Results`][ropt.results.Results] object represents the outcome of the
calculation for a **single variable vector** — that is, the objective and
gradient values computed at one point in variable space. However, the optimizer
may request evaluations at multiple variable vectors in a single batch (e.g.,
multiple perturbations or multiple candidates in a gradient-free method). In
that case, the event payload contains a *sequence* of `Results` objects, one per
variable vector evaluated in that batch.

Two concrete subclasses exist:

- [`FunctionResults`][ropt.results.FunctionResults] — objective and constraint
  values for a batch.
- [`GradientResults`][ropt.results.GradientResults] — gradient estimates for
  the objective and constraints.

Each carries nested [`ResultField`][ropt.results.ResultField] objects:

| Result             | Fields                                                                                          |
| ------------------ | ----------------------------------------------------------------------------------------------- |
| `FunctionResults`  | `evaluations` ([`FunctionEvaluations`][ropt.results.FunctionEvaluations]), `functions` ([`Functions`][ropt.results.Functions]), `realizations` ([`Realizations`][ropt.results.Realizations]), `constraint_info` ([`ConstraintInfo`][ropt.results.ConstraintInfo]). |
| `GradientResults`  | `evaluations` ([`GradientEvaluations`][ropt.results.GradientEvaluations]), `gradients` ([`Gradients`][ropt.results.Gradients]). |

### What each field holds

#### `FunctionResults` fields

- **`evaluations`** ([`FunctionEvaluations`][ropt.results.FunctionEvaluations])
  — the raw per-realization evaluation data:
    - `variables`: the unperturbed variable vector, shape $(n_v,)$.
    - `objectives`: objective values per realization, shape $(n_r, n_o)$.
    - `constraints`: constraint values per realization, shape $(n_r, n_c)$
      (only present when nonlinear constraints are configured).
    - `metadata`: optional dict of per-realization metadata arrays, each of
      shape $(n_r,)$.
- **`functions`** ([`Functions`][ropt.results.Functions]) — aggregated values
  derived from the per-realization evaluations (or `None` if all realizations
  failed):
    - `target_objective`: the single weighted scalar the optimizer minimizes
      (0-D array).
    - `objectives`: individual objective values, shape $(n_o,)$.
    - `constraints`: individual constraint values, shape $(n_c,)$ (if
      configured).
- **`realizations`** ([`Realizations`][ropt.results.Realizations]) — ensemble
  metadata:
    - `evaluated_realizations`: boolean array indicating which realizations were
      evaluated, shape $(n_r,)$.
    - `objective_weights`: per-realization objective weights, shape
      $(n_o, n_r)$. May change during optimization (e.g., when realization
      filters are active).
    - `constraint_weights`: per-realization constraint weights, shape
      $(n_c, n_r)$ (if constraints are configured).
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

- **`evaluations`** ([`GradientEvaluations`][ropt.results.GradientEvaluations])
  — evaluation data for perturbed variables:
    - `variables`: the unperturbed variable vector, shape $(n_v,)$.
    - `perturbed_variables`: perturbed variable values, shape
      $(n_r, n_p, n_v)$.
    - `perturbed_objectives`: objective values for each perturbation, shape
      $(n_r, n_p, n_o)$.
    - `perturbed_constraints`: constraint values for each perturbation, shape
      $(n_r, n_p, n_c)$ (if configured).
    - `metadata`: optional dict of per-realization/perturbation metadata
      arrays, each of shape $(n_r, n_p)$.
- **`gradients`** ([`Gradients`][ropt.results.Gradients]) — aggregated gradient
  values (or `None` if estimation failed):
    - `target_objective`: gradient of the weighted objective w.r.t. each
      variable, shape $(n_v,)$.
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
- **`names`**: a mapping from [`AxisName`][ropt.enums.AxisName] values to label
  tuples. Used to produce labelled multi-index DataFrames when exporting (see
  [Exporting to pandas](#exporting-to-pandas)).

## Accessing result data

Common access patterns:

```python
result.evaluations.variables       # variable vector(s) evaluated
result.functions.target_objective  # weighted scalar objective
result.functions.objectives        # per-objective values (after weighting)
result.functions.constraints       # per-constraint values
```

If `functions` is `None`, the result represents a request that produced no
valid values (e.g. all realizations failed). Always guard accesses:

```python
if result.functions is not None:
    print(result.functions.target_objective)
```

## Axes and dimensionality

Much of the data within result objects is multi-dimensional. For example, the
`objectives` field within
[`FunctionEvaluations`][ropt.results.FunctionEvaluations] is a 2-D array where
each row is a realization and each column is an objective.

To simplify exporting and reporting, the identity of each dimension is stored as
axis metadata on each field. The [`ResultField`][ropt.results.ResultField] base
class provides a [`get_axes`][ropt.results.ResultField.get_axes] class method
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

!!! note
    Dimensionality is fixed: even with a single objective, result arrays still
    include an `OBJECTIVE` axis of length one.

## Domain transforms on results

When [transforms](transforms.md) are configured, optimization internally
operates in the *optimizer domain* — variables, objectives, and constraints may
be scaled or shifted for numerical stability. Results attached to events are in
this optimizer domain.

The [`transform_from_optimizer`][ropt.results.Results.transform_from_optimizer]
method reverses these transforms, mapping results back to the *user domain*.

In the [simple API](../running/running.md), results are always transformed to the user
domain automatically.

In [workflows](../workflows/workflows.md), event handlers determine how results are returned,
for instance by offering a `domain` argument that controls whether results are
handled in user or optimizer domain. See [Optimization Workflows](../workflows/workflows.md)
for details on how individual event handlers handle this.

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

    The `to_dataframe` and `results_to_dataframe` functions shown here are the
    low-level export primitives. Most users do not call them directly: the
    [`DataFrameHandler`](../running/running.md#dataframehandler) builds and updates
    these tables automatically as an optimization runs. This section explains
    what that handler produces under the hood.

The row index and the unstacked column labels come from the
[`names`](configuration.md#names) mapping in the configuration. If an axis is
not named, its labels fall back to 0-based integer indices. For example,
exporting the objectives of a single result **without** any `names` gives plain
numbers for both the realization and the objective axes:

```python
df = result.to_dataframe("evaluations", select=["objectives"])
```

```
                               objectives
batch_id realization objective
1        0           0               2.10
                     1               0.94
         1           0               2.35
                     1               1.02
```

Adding a `names` entry replaces those numbers with meaningful labels. The
examples below assume the realizations are named `"r0"`/`"r1"` and the objectives
`"val"`/`"cost"`.

### Exporting a single result field

The [`to_dataframe`][ropt.results.Results.to_dataframe] method on an individual
result exports one field (or a subset of its sub-fields):

```python
df = result.to_dataframe("evaluations", select=["variables", "objectives"])
```

By default, every axis of the exported sub-fields becomes a level in a
multi-index. For example, `objectives` in
[`FunctionEvaluations`][ropt.results.FunctionEvaluations] has the axes
`REALIZATION` and `OBJECTIVE`, so exporting it keeps both in the index — now
with the configured names:

```python
df = result.to_dataframe("evaluations", select=["objectives"])
```

```
                               objectives
batch_id realization objective
1        r0          val             2.10
                     cost            0.94
         r1          val             2.35
                     cost            1.02
```

Passing `unstack` pivots selected axes out of the index and into columns. Here
the `OBJECTIVE` axis is unstacked:

```python
from ropt.enums import AxisName

df = result.to_dataframe(
    "evaluations",
    select=["objectives"],
    unstack=[AxisName.OBJECTIVE],
)
```

```
                     (objectives, val)  (objectives, cost)
batch_id realization
1        r0                       2.10                0.94
         r1                       2.35                1.02
```

The unstacked axis is flattened into the column labels, so each new column is a
tuple of the sub-field name and the axis label — here `("objectives", "val")`
and `("objectives", "cost")`. Unstacking more axes adds more elements to these
tuples; unstacking every axis leaves a flat table with one row per result.

### Aggregating multiple results

[`results_to_dataframe`][ropt.results.results_to_dataframe] builds on
`to_dataframe` to convert a *sequence* of results into a single DataFrame, one
row per result. It automatically unstacks the most common axes (`VARIABLE`,
`OBJECTIVE`, `NONLINEAR_CONSTRAINT`) into columns:

```python
from ropt.results import results_to_dataframe

df = results_to_dataframe(
    all_results,
    fields={"evaluations.variables"},
    result_type="functions",
)
```

```
          (evaluations.variables, x0)  (evaluations.variables, x1)  (evaluations.variables, x2)
batch_id
1                                0.30                         0.42                        -0.11
2                                0.55                         0.48                         0.02
3                                0.61                         0.50                         0.10
```

Each column is a `(field, label)` pair, and each row is one result identified by
its `batch_id`. Field names use dot notation for nested sub-fields (e.g.,
`evaluations.variables`, `functions.target_objective`). The `result_type`
argument selects which results to process: `"functions"` for
[`FunctionResults`][ropt.results.FunctionResults] only, `"gradients"` for
[`GradientResults`][ropt.results.GradientResults] only.

### Metadata columns

The two kinds of [metadata](#metadata) are exported differently, and the two
functions are **not** symmetric.

`to_dataframe` works on a single result field, so it reaches that field's
**per-realization metadata** (named `metadata.<key>`), which keeps the
`realization` axis. For example, if the objective attached a per-realization
`shift`:

```python
df = result.to_dataframe("evaluations", select=["metadata.shift"])
```

```
                     metadata.shift
batch_id realization
1        r0                     0.9
         r1                     1.1
```

`to_dataframe` **cannot** reach the run-level **result metadata**, because it is
not part of any single field. Use `results_to_dataframe` for that: name it with a
top-level `metadata.` prefix to get one value per result — handy for pulling in a
run tag:

```python
df = results_to_dataframe(
    all_results,
    fields={"metadata.run_id", "functions.target_objective"},
    result_type="functions",
)
```

```
          functions.target_objective  metadata.run_id
batch_id
1                               1.83                0
2                               0.42                1
```

### Labels and the index

Every axis of an exported field becomes an index level, named after its
[`AxisName`][ropt.enums.AxisName] value (e.g. `"variable"`, `"realization"`,
`"objective"`), and `batch_id` is always prepended so results from different
batches stay distinct. The label on each level — and on each unstacked column —
comes from the [`names`](configuration.md#names) mapping in the configuration, a
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
differences you need to know about.

**Index levels become ordinary columns.** What pandas puts in the index, polars
puts in leading columns of the frame:

```python
df = result.to_polars("evaluations", select=["objectives"])
```

```
┌──────────┬─────────────┬───────────┬────────────┐
│ batch_id ┆ realization ┆ objective ┆ objectives │
╞══════════╪═════════════╪═══════════╪════════════╡
│ 1        ┆ r0          ┆ val       ┆ 2.10       │
│ 1        ┆ r0          ┆ cost      ┆ 0.94       │
│ 1        ┆ r1          ┆ val       ┆ 2.35       │
│ 1        ┆ r1          ┆ cost      ┆ 1.02       │
└──────────┴─────────────┴───────────┴────────────┘
```

**Tuple column labels become joined strings.** Where pandas produces the column
`("objectives", "val")`, polars produces `"objectives,val"`. The separator is
configurable with the `sep` argument, which defaults to `","`:

```python
from ropt.enums import AxisName

df = result.to_polars(
    "evaluations",
    select=["objectives"],
    unstack=[AxisName.OBJECTIVE],
)
```

```
┌──────────┬─────────────┬────────────────┬─────────────────┐
│ batch_id ┆ realization ┆ objectives,val ┆ objectives,cost │
╞══════════╪═════════════╪════════════════╪═════════════════╡
│ 1        ┆ r0          ┆ 2.10           ┆ 0.94            │
│ 1        ┆ r1          ┆ 2.35           ┆ 1.02            │
└──────────┴─────────────┴────────────────┴─────────────────┘
```

Aggregating a sequence of results works the same way:

```python
from ropt.results import results_to_polars

df = results_to_polars(
    all_results,
    fields={"evaluations.variables"},
    result_type="functions",
)
```

```
┌──────────┬──────────────────────────┬──────────────────────────┬──────────────────────────┐
│ batch_id ┆ evaluations.variables,x0 ┆ evaluations.variables,x1 ┆ evaluations.variables,x2 │
╞══════════╪══════════════════════════╪══════════════════════════╪══════════════════════════╡
│ 1        ┆ 0.30                     ┆ 0.42                     ┆ -0.11                    │
│ 2        ┆ 0.55                     ┆ 0.48                     ┆ 0.02                     │
│ 3        ┆ 0.61                     ┆ 0.50                     ┆ 0.10                     │
└──────────┴──────────────────────────┴──────────────────────────┴──────────────────────────┘
```

Metadata behaves exactly as described [above](#metadata-columns): per-realization
metadata is reachable from `to_polars`, run-level result metadata only from
`results_to_polars`:

```python
df = results_to_polars(
    all_results,
    fields={"metadata.run_id", "functions.target_objective"},
    result_type="functions",
)
```

```
┌──────────┬────────────────────────────┬─────────────────┐
│ batch_id ┆ functions.target_objective ┆ metadata.run_id │
╞══════════╪════════════════════════════╪═════════════════╡
│ 1        ┆ 1.83                       ┆ 0               │
│ 2        ┆ 0.42                       ┆ 1               │
│ 3        ┆ 0.11                       ┆ 2               │
└──────────┴────────────────────────────┴─────────────────┘
```

!!! note

    Because polars keeps the keys as real columns, it can join fields that vary
    at different granularities — for example a per-batch gradient with
    per-perturbation evaluations — by repeating the coarser values across the
    finer rows. Pandas cannot align such fields and returns them as disjoint
    blocks of rows padded with missing values instead, so prefer polars when a
    single table has to mix granularities.

## Where to next

- Run an optimization and receive results via callbacks:
  [Deterministic Optimization](../getting_started/deterministic.md).
- Use event handlers to collect or react to results in a workflow:
  [Optimization Workflows](../workflows/workflows.md).
