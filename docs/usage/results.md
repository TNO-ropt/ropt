# Working with Results

`ropt` exposes the full intermediate and final state of an optimization through
[`Results`][ropt.results.Results] objects. This page describes the result
classes and how to inspect them; see [Basic Optimization](basic.md) and
[Optimization Workflows](workflows.md) for how results are produced and
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
    - `active_realizations`: boolean array indicating which realizations were
      evaluated, shape $(n_r,)$.
    - `failed_realizations`: boolean array indicating failures, shape $(n_r,)$.
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

- **`batch_id`**: an optional integer identifying the evaluation batch
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

When using [`BasicOptimizer`][ropt.workflow.BasicOptimizer], results passed to
observer callbacks are always transformed to user domain automatically.

In [workflows](workflows.md), event handlers determine how results are returned,
for instance by offering a `domain` argument that controls whether results are
handled in user or optimizer domain. See [Optimization Workflows](workflows.md)
for details on how individual event handlers handle this.

## Exporting to pandas

`ropt` can export results to `pandas` DataFrames for analysis and reporting.
This requires the `pandas` optional extra (see [Installation](installation.md)).

### Exporting a single result field

The [`to_dataframe`][ropt.results.Results.to_dataframe] method on an individual
result exports one field (or a subset of its sub-fields):

```python
from ropt.enums import AxisName

df = result.to_dataframe(
    "evaluations",
    select=["variables", "objectives"],
)
```

By default, all axes of the exported sub-fields are represented as levels in a
multi-index. For example, exporting `objectives` from
[`FunctionEvaluations`][ropt.results.FunctionEvaluations] (which has axes
`REALIZATION` and `OBJECTIVE`) produces a DataFrame with a two-level index: one
level for the realization, one for the objective.

You can *unstack* selected axes into separate columns using the `unstack`
argument:

```python
df = result.to_dataframe(
    "evaluations",
    select=["variables", "objectives"],
    unstack=[AxisName.VARIABLE, AxisName.OBJECTIVE],
)
```

This pivots the specified axes out of the index and into columns. Column names
become tuples combining the sub-field name and the axis label(s). For instance,
unstacking `VARIABLE` on a `variables` sub-field with labels `("x0", "x1",
"x2")` produces columns named `("variables", "x0")`, `("variables", "x1")`,
`("variables", "x2")`.

### Aggregating multiple results

[`results_to_dataframe`][ropt.results.results_to_dataframe] builds on
`to_dataframe` to convert a *sequence* of results into a single DataFrame. It
handles field selection and automatically unstacks the most common axes into
columns (e.g., `VARIABLE`, `OBJECTIVE`, `NONLINEAR_CONSTRAINT`):

```python
from ropt.results import results_to_dataframe

df = results_to_dataframe(
    all_results,
    fields={"evaluations.variables", "functions.objectives"},
    result_type="functions",
)
```

Field names use dot notation for nested sub-fields (e.g.,
`evaluations.variables`, `functions.target_objective`). For `metadata`
dictionaries, use a second dot: `evaluations.metadata.my_key`.

The `result_type` argument selects which result objects to process:
`"functions"` includes only
[`FunctionResults`][ropt.results.FunctionResults], `"gradients"` includes only
[`GradientResults`][ropt.results.GradientResults].

### Multi-index labelling

The DataFrame index is constructed from the axis metadata of each exported
field. Each axis becomes an index level, named after the
[`AxisName`][ropt.enums.AxisName] value (e.g., `"variable"`, `"realization"`,
`"objective"`).

Labels for each level come from the `names` attribute on the result (which is
populated from the [`names`](configuration.md#names) field in
[`EnOptContext`][ropt.context.EnOptContext]). This is a dict mapping axis name
strings to tuples of labels:

```python
CONFIG = {
    ...
    "names": {
        "variable": ("x0", "x1", "x2"),
        "objective": ("NPV", "cost"),
    },
}
```

When labels are provided for an axis, they appear in the DataFrame index or
column headers. When no labels are provided, 0-based integer indices are used
instead.

If `batch_id` is set on the result, a `"batch_id"` level is prepended to the
index, allowing results from multiple batches to be distinguished in an
aggregated DataFrame.

## Where to next

- Run an optimization and receive results via callbacks:
  [Basic Optimization](basic.md).
- Use event handlers to collect or react to results in a workflow:
  [Optimization Workflows](workflows.md).
