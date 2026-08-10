# Running Optimizations

!!! note

    This is one of two ways to **run** an optimization; the other is
    [Optimization Workflows](../workflows/workflows.md). What the optimization
    does — its variables, objectives, constraints, and components — is set up in
    [Optimizer Setup](../optimizer_setup/key_concepts.md), the same whichever way you run
    it.

The `ropt.simple` module is the recommended entry point for running an
optimization. Everything you need is imported from a single module:

```python
from ropt.simple import optimize
```

You give `optimize` three things:

- a **config** dictionary that describes the problem,
- a **start point** (the first set of variable values),
- an **objective function** that returns the value to minimize.

```python
import numpy as np
from ropt.simple import optimize

config = {"variables": {"variable_count": 3, "perturbation_magnitudes": 1e-6}}


def objective(variables, context):
    return float(np.sum((variables - 1.0) ** 2))


result = optimize(config, np.zeros(3), objective)
print(result.variables)          # the best variables found
print(result.target_objective)   # the objective value there
```

That is the basic pattern. The rest of this page explains each part and the
additional options.

## The objective function

The objective is a Python function with two arguments:

```python
from ropt.simple import EvaluationFunctionContext


def objective(variables: np.ndarray, context: EvaluationFunctionContext) -> float:
    ...
```

- `variables` is a 1-D NumPy array: one set of variable values to evaluate.
- `context` tells you *which* evaluation this is. Its most important field is
  `context.realization`, the realization number (see
  [Ensembles](#optimizing-over-an-ensemble) below).

The function returns the objective value. There are three ways to return it:

- a **single number** when there is one objective and no nonlinear constraints;
- a **list** of numbers when there are several objectives or nonlinear
  constraints — put the objectives first, then the constraints;
- an [`EvaluationFunctionResult`][ropt.components.evaluators.EvaluationFunctionResult]
  when you also want to attach `metadata`.

If a realization fails to compute, return `float("nan")` for it. `ropt` treats
`NaN` as a failed realization and keeps going, as long as enough realizations
succeed (see [Configuration](../optimizer_setup/configuration.md)).

## The result

`optimize` returns an [`OptimizeResult`][ropt.simple.OptimizeResult]:

```python
result = optimize(config, x0, objective)

result.exit_code         # why the run stopped (an ropt.enums.ExitCode)
result.variables         # the best variables, or None if none was valid
result.target_objective  # the objective value at the best point, or None
result.objectives        # the separate objective values, or None
result.constraints        # the nonlinear constraint values, or None
result.results           # the full low-level result object (see below)
```

The fields are `None` when the run produced no valid result (for example when
too few realizations succeeded). `result.results` is the full
[`FunctionResults`][ropt.results.FunctionResults] object; you rarely need it at
first, but it holds every detail if you do. See
[Working with Results](../optimizer_setup/results.md).

## Reporting progress

Pass a `report` callback to watch the optimization as it runs. It is called once
for every function evaluation, with an [`EvaluateResult`][ropt.simple.EvaluateResult]:

```python
def report(result):
    print(result.target_objective)


optimize(config, x0, objective, report=report)
```

## Attaching metadata

You can attach arbitrary **metadata** to a run, from two sources:

- **Constant, per run** — pass a `metadata` dict to `optimize`, `optimize_many`,
  `evaluate`, or `evaluate_many`. `ropt` copies it onto every result the run
  produces, which is handy for tagging a run:

  ```python
  result = optimize(config, x0, objective, metadata={"run_id": 7})
  print(result.results.metadata["run_id"])   # 7
  ```

  With `optimize_many`, give one dict (shared by all runs) or a list with one
  dict per run:

  ```python
  results = optimize_many(
      config,
      start_points,
      objective,
      metadata=[{"run_id": i} for i in range(len(start_points))],
  )
  ```

- **Per evaluation** — return an
  [`EvaluationFunctionResult`][ropt.components.evaluators.EvaluationFunctionResult]
  from the objective with a `metadata` field. This value is stored per
  realization, next to the objective values:

  ```python
  from ropt.simple import EvaluationFunctionResult

  def objective(variables, context):
      value = ...
      return EvaluationFunctionResult(objectives=value, metadata={"seconds": 1.3})
  ```

Neither kind is interpreted by `ropt`. Constant metadata ends up on
`result.results.metadata`; per-evaluation metadata on
`result.results.evaluations.metadata` (one entry per realization). See
[Working with Results](../optimizer_setup/results.md#metadata) for how
each appears in the pandas export. The full runnable script is
[examples/simple/metadata.py](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/metadata.py).

## Evaluating without optimizing

Sometimes you only want the objective value for a point, without running an
optimizer. Use [`evaluate`][ropt.simple.evaluate] for one point and
[`evaluate_many`][ropt.simple.evaluate_many] for several:

```python
from ropt.simple import evaluate, evaluate_many

single = evaluate(config, x, objective)             # one EvaluateResult
batch = evaluate_many(config, matrix, objective)    # one per row of the matrix
```

An [`EvaluateResult`][ropt.simple.EvaluateResult] has the same fields as
`OptimizeResult`, minus `exit_code` and `variables` (you supplied the point
yourself; it is still on `result.results.evaluations.variables`).

## Optimizing over an ensemble

Many problems are uncertain: the objective depends on parameters that vary
across a set of *realizations*. You add a `realizations` section to the config
and use `context.realization` to pick the right parameters:

```python
config = {
    "variables": {"variable_count": 3, "perturbation_magnitudes": 1e-6},
    "realizations": {"weights": [1.0] * 10},   # ten realizations
}


def objective(variables, context):
    a = uncertain_parameters[context.realization]
    return float(np.sum((variables - a) ** 2))
```

`optimize` then minimizes the weighted average objective over all realizations.

## Running in parallel

By default `optimize` runs on the calling thread, one evaluation at a time. To
run the evaluations in parallel, open an execution block first. See
[Running in Parallel](../getting_started/execution.md) for a full explanation of the three choices
and their trade-offs:

```python
from ropt.simple import processes

with processes(workers=8):
    result = optimize(config, x0, objective)
```

### Running on an HPC cluster

The [`hpc`][ropt.simple.hpc] block submits each evaluation as a job to an HPC
queue (through `pysqa`); it needs the `ropt[hpc]` extra. With no further
arguments it uses the default cluster and queue from the `pysqa` configuration of
your `ropt` installation:

```python
from ropt.simple import hpc

with hpc(workers=10):
    result = optimize(config, x0, objective)
```

`hpc` accepts the following parameters:

| Parameter     | Description                                                                |
| ------------- | ------------------------------------------------------------------------- |
| `workers`     | Maximum number of concurrent cluster jobs (default: 1).                   |
| `cores`       | Number of CPUs per job (default: 1).                                      |
| `cluster`     | Cluster name, when the `pysqa` config defines several.                    |
| `queue`       | Queue or partition name.                                                  |
| `workdir`     | Shared-filesystem working directory (defaults to the current directory).  |
| `config_path` | Path to the `pysqa` configuration directory.                              |
| `template`    | Inline submission-script template, used instead of a config.              |
| `queue_type`  | Queueing system type (default: `"slurm"`).                                |

## Many optimizations at once

To run several optimizations together, use
[`optimize_many`][ropt.simple.optimize_many] inside an execution block. Any of
`config`, `x0`, or `objective` may be a single value (used for every run) or a
list (one per run):

```python
from ropt.simple import optimize_many, threads

with threads(workers=4):
    results = optimize_many(config, start_points, objective)   # one run per start
```

There are two independent levels of concurrency here:

- **The optimizations** always run concurrently, each on its own driver thread.
  This is built into `optimize_many` and does not depend on which block you open;
  the `limit` argument caps how many run at the same time.
- **The function evaluations** inside those runs share the block's single worker
  pool, and the block decides how they are parallelized. With `threads(workers=1)`
  (the default worker count) the runs still progress together, but their
  evaluations are executed one at a time. A larger pool — `threads(workers=n)`,
  `processes`, or `hpc` — runs several evaluations at once.

An execution block is required: calling `optimize_many` without one raises a
`RuntimeError`.

## Result handlers

An [`optimize`][ropt.simple.optimize] call returns only the best result. A
**handler** lets you collect or react to *every* result instead: it is an object
that observes an optimization and processes its results as they arrive — keeping
them, tabulating them, or invoking a callback.

Attach handlers to a single optimization with the `handlers` argument, or share
them across a block of runs with a [`handlers`][ropt.simple.handlers] block so
one handler accumulates results from every `optimize` inside it:

```python
from ropt.simple import HistoryHandler, handlers, optimize

history = HistoryHandler()

optimize(config, x0, objective, handlers=[history])   # a single run
# ...or accumulate across many runs:
with handlers(history):
    for x0 in start_points:
        optimize(config, x0, objective)

print(history.results)   # every result collected
```

Handlers that store results expose them through `handler["results"]` (and, for
`HistoryHandler`, the `history.results` shortcut).

### Built-in handlers

`ropt` ships several ready-to-use handlers, all re-exported from `ropt.simple`.

#### `ResultsHandler`

[`ResultsHandler`][ropt.simple.ResultsHandler] keeps a single result, read via
`handler["results"]`:

- `what="best"` (default) keeps the result with the lowest weighted objective
  seen so far; `what="last"` keeps the most recent valid result.
- `constraint_tolerance` (optional) discards results that violate a constraint
  by more than the given tolerance.
- `filter` (optional) is a callable that receives each
  [`Results`][ropt.results.Results] and returns `True` to keep it or `False` to
  drop it.

#### `HistoryHandler`

[`HistoryHandler`][ropt.simple.HistoryHandler] keeps *every* result it receives,
in order, as a tuple read via `handler["results"]` (or `handler.results`). It is
`None` until the first result arrives.

#### `DataFrameHandler`

[`DataFrameHandler`][ropt.simple.DataFrameHandler] collects results into named
pandas DataFrames (`pandas` must be installed). Define a table with
`add_table(name, table_type, columns)`, where `table_type` is `"functions"` or
`"gradients"` and `columns` maps result-field names (dotted attribute syntax) to
column titles:

```python
from ropt.simple import DataFrameHandler

tables = DataFrameHandler()
tables.add_table(
    "summary",
    "functions",
    {
        "batch_id": "Batch",
        "functions.objectives": "Objective",
        "evaluations.variables": "Variable",
    },
)
optimize(config, x0, objective, handlers=[tables])
df = tables["summary"]
```

Read one table with `tables["summary"]`, or all of them with `get_tables()`. A
field whose value is a vector or matrix expands to several columns; the extra
column levels come from the field's axis labels (or indices), joined to the
title with a separator (`,` by default, set with `sep=`). For example, a
length-2 `evaluations.variables` gives `Variable,v0` and `Variable,v1`. Because
the column names follow
[`results_to_dataframe`](../optimizer_setup/results.md#metadata-columns),
both result-level and per-realization metadata can be included and renamed.

Convenience methods:

- `set_default_tables()` registers a standard set of tables (`functions`,
  `evaluations`, `constraints` for function results; `gradients`,
  `perturbations` for gradient results).
- `add_column(table, name, title)` adds one column to an existing table.
- `set_callback(fn)` calls `fn(event)` whenever the tables are updated.

### Custom handlers

Handlers are not limited to the built-ins: you — or another package — can
provide your own by implementing `ropt`'s event-handler interface. The
[Low-Level API](../workflows/workflows.md#event-handlers) describes the event
model, the handler protocol, and how to write one.

## A note on enums

A few config values and result fields use enumerations, such as
[`VariableType`][ropt.enums.VariableType] for integer variables and
[`ExitCode`][ropt.enums.ExitCode] for `result.exit_code`. These are **not** part
of `ropt.simple`; import them from [`ropt.enums`][ropt.enums]:

```python
from ropt.enums import ExitCode, VariableType
```
