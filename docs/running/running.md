# Running Optimizations

!!! note
    This page uses the module-level functions [`optimize`][ropt.optimize] and
    [`evaluate`][ropt.evaluate], which evaluate one point at a time and return
    when they have finished. Parallel evaluation, several optimizations at once and
    aborting from outside need a [session](sessions.md). Everything this page
    explains applies there unchanged.

Everything needed to run an optimization is imported from the `ropt` package
itself:

```python
from ropt import optimize
```

You give [`optimize`][ropt.optimize] three things:

- a **config** dictionary that describes the problem — its variables,
  objectives, constraints and components, covered in
  [Optimizer Setup](../optimizer_setup/key_concepts.md),
- a **start point** (the first set of variable values),
- an **evaluation function** that returns the objective value(s) to minimize,
  followed by any constraint values.

```python
import numpy as np
from ropt import optimize

config = {"variables": {"variable_count": 3, "perturbation_magnitudes": 1e-6}}

def objective(variables, context):
    return float(np.sum((variables - 1.0) ** 2))

result = optimize(config, np.zeros(3), objective)
if result.results is not None:
    print(result.results.variables)         # the best variables found
    print(result.results.target_objective)  # the objective value there
```

That is the basic pattern. The rest of this page explains each part and the
additional options.

## The evaluation function

The evaluation function is a Python function with two arguments:

```python
from ropt import EvaluationFunctionContext


def objective(variables: np.ndarray, context: EvaluationFunctionContext) -> float:
    ...
```

- `variables` is a 1-D NumPy array: one set of variable values to evaluate.
- `context` is an
  [`EvaluationFunctionContext`][ropt.EvaluationFunctionContext]
  that identifies *which* evaluation this is:
    - `context.realization` — the realization number, for a problem with an
      ensemble of realizations; `optimize` then minimizes a robust objective
      over all of them. See [Ensemble-Based
      Optimization](../getting_started/ensemble.md).
    - `context.perturbation` — the perturbation number, counted from 0, for an
      evaluation at a perturbed point; `-1` for a function evaluation. `ropt`
      evaluates perturbed points to estimate a
      [gradient](../optimizer_setup/gradients.md).
    - `context.function_id` — the function evaluation number within the batch,
      counted from 0; `-1` for an evaluation at a perturbed point.
    - `context.batch_id` — the batch number. `ropt` requests evaluations in
      batches, and each batch in a program gets its own number.
    - `context.row_idx` — the position of the evaluation within its batch,
      counted from 0.
    - `context.metadata` — the `metadata` dict the optimization was started
      with, if any (see [Attaching metadata](#attaching-metadata)).

The function returns the objective value(s), followed by any constraint values.
There are three ways to return them:

- a **single number** when there is one objective and no nonlinear constraints;
- a **numpy vector** or a **sequence** of numbers when there are several
  objectives or nonlinear constraints — put the objectives first, then the
  constraints;
- an [`EvaluationFunctionResult`][ropt.EvaluationFunctionResult]
  when you also want to attach `metadata`; it holds `objectives`, `constraints`
  and `metadata` in separate fields.

If a realization fails to compute, return `float("nan")` or `numpy.nan` for it.
`ropt` treats `NaN` as a failed realization and keeps going, as long as enough
realizations succeed. How many is enough is set by
[`realization_min_success`](../optimizer_setup/configuration_sections.md#realizations),
which defaults to *all* of them — so a single `NaN` ends the optimization with
`TOO_FEW_REALIZATIONS` unless you lower it.

### Running an external program

When the evaluation function runs a an external program, start it with
[`run_program`][ropt.run_program]. This is the supported way to run external
programs from `ropt`. It takes the program and its arguments as a list, waits
for the program to exit, and returns its exit status:

```python
from ropt import run_program


def objective(variables, context):
    status = run_program(["my_simulator", "input.json"], cwd=workdir)
    if status != 0:
        return float("nan")
    return read_objective(workdir)
```

`run_program` terminates the program, together with every program it started,
when the optimization is stopped. This matters most when evaluations run in
parallel or several optimizations run at once: an abort or a failure then stops
every program that is still running. A program started with the `subprocess.run`
function from the Python standard library keeps running after an abort, so its
use is not recommended. See [Running External Programs](programs.md).

## The result

`optimize` returns an [`OptimizationResult`][ropt.OptimizationResult],
which carries two things:

```python
result = optimize(config, x0, objective)

result.exit_code  # why the optimization stopped (an ExitCode)
result.results    # the best evaluation, or None if none was valid
```

`results` is a [`FunctionResults`][ropt.results.FunctionResults], so the best
point and its values are read from it at the paths described in [Working with
Results](../results/results.md):

```python
if result.results is not None:
    result.results.variables             # the best variables
    result.results.target_objective      # the objective value there
    result.results.functions.objectives  # the separate objective values
    result.results.functions.constraints # the nonlinear constraint values
```

`results` is `None` when the optimization produced no valid result: too few
realizations
succeeded, or no result ever satisfied the constraints — see
[When something goes wrong](#when-something-goes-wrong). One check therefore
covers every field.

## Reporting progress

Pass a `report` callback to track the optimization as it runs. It is called once
for every function evaluation, with the
[`FunctionResults`][ropt.results.FunctionResults] of that evaluation:

```python
def report(result):
    print(result.variables, result.target_objective)


optimize(config, x0, objective, report=report)
```

To store results rather than print them, pass a **handler** with the `handlers`
argument. [`HistoryHandler`][ropt.HistoryHandler] keeps every result, and
[`DataFrameHandler`][ropt.DataFrameHandler] collects selected fields into
`polars` or `pandas` tables. See [Result Handlers](../results/handlers.md).

### Reporting gradients as well

Set `report_gradients=True` and the callback is also called with each
[`GradientResults`][ropt.results.GradientResults], which arrives in its own
evaluation. The callback then receives both kinds and has to tell them apart:

```python
from ropt import FunctionResults


def report(result):
    if isinstance(result, FunctionResults):
        print("objective", result.target_objective)
    else:
        print("gradient", result.target_gradient)


optimize(config, x0, objective, report=report, report_gradients=True)
```

Gradients are generally paired with a function evaluation at the same point. A
[`GradientResults`][ropt.results.GradientResults] records an ID of that function
evaluation. The [`uses`][ropt.results.GradientResults.uses] method can be used
to check if a [`FunctionResults`][ropt.results.FunctionResults] object has the
same ID.

### Stopping early from the callback

The `report` callback doubles as a **user-defined stopping criterion**: return
`True` and the optimization stops gracefully after the current evaluation, with
exit code `STOPPED`. Any other return value (including `None`) lets it
continue.

```python
from ropt import ExitCode


def report(result):
    if result.target_objective is not None and result.target_objective < 1e-6:
        return True  # good enough — stop this optimization
    return None


result = optimize(config, x0, objective, report=report)
assert result.exit_code is ExitCode.STOPPED
```

When [running multiple optimizations at once](many_runs.md), this stops only the
optimization whose callback returned `True`; the others continue. To abort all
running optimizations in a session, call [`Session.abort`][ropt.Session.abort].
For its effect when several optimizations run together, see [Aborting from
outside](many_runs.md#stopping-from-outside).

The runnable script is
[examples/stopping.py](https://github.com/TNO-ropt/ropt/blob/main/examples/stopping.py),
which stops an optimization after a fixed number of results and then reads the
best point it had reached.


## Attaching metadata

You can attach arbitrary **metadata** to an optimization or evaluation, from two
sources:

- **Constant, per task** — pass a `metadata` dict to `optimize`, `evaluate`, or
  `evaluate_batch`. `ropt` copies it onto every result the task produces, which
  is one way to tag an optimization:

  ```python
  result = optimize(config, x0, objective, metadata={"id": 7})
  print(result.results.metadata["id"])   # 7
  ```

  The same dict also reaches the evaluation function itself, as
  `context.metadata` — useful when the evaluation needs to know which
  optimization it is part of. With several optimizations at once, this is how
  they are told apart; see [Give each optimization an ID](many_runs.md).

- **Per evaluation** — return an
  [`EvaluationFunctionResult`][ropt.EvaluationFunctionResult]
  from the evaluation function with a `metadata` field. This value is stored per
  realization, next to the objective values:

  ```python
  from ropt import EvaluationFunctionResult

  def objective(variables, context):
      value = ...
      residual = variables - x0
      return EvaluationFunctionResult(objectives=value, metadata={"residual": residual})

  result = optimize(config, x0, objective)
  print(result.results.evaluations.metadata["residual"].shape)  # (3, 3)
  ```

  A key does not have to be set by every realization; those that do not set it
  get `np.nan` for numeric values and `None` otherwise, so a numeric column with
  missing realizations is widened to `float64` — numpy has no integer NaN. A
  key set by every realization keeps its natural dtype, and mixing strings and
  non-strings under one key raises a `ValueError`.

  Returning an array instead of a scalar gives the key its own
  [user-defined axis](../results/results.md#user-defined-axes), which
  the `names` section of the configuration can label and which the
  [`DataFrameHandler`](../results/handlers.md#dataframehandler) spreads over one
  column per entry.

  In the example above, the first axis is the realization; the second is the
  `residual` user-defined axis. The
  [`DataFrameHandler`](../results/handlers.md#dataframehandler) example shows
  how to label and export that axis.

Neither kind is interpreted by `ropt`. Constant metadata ends up on
`result.results.metadata`; per-evaluation metadata on
`result.results.evaluations.metadata` (one entry per realization). Both kinds
can be tabulated as columns by the
[`DataFrameHandler`](../results/handlers.md#dataframehandler). See [Working with
Results](../results/results.md#metadata) for how each appears. The full runnable
script is
[examples/metadata.py](https://github.com/TNO-ropt/ropt/blob/main/examples/metadata.py).

## Evaluating without optimizing

Sometimes you only want the objective value for a point, without running an
optimizer. Use [`evaluate`][ropt.evaluate] for one point and
[`evaluate_batch`][ropt.evaluate_batch] for several:

```python
from ropt import evaluate, evaluate_batch

single = evaluate(config, x, objective)
batch = evaluate_batch(config, matrix, objective)

single.results  # one FunctionResults
batch.results   # one per row of the matrix
```

Both return an [`EvaluationResult`][ropt.EvaluationResult], whose `results`
holds what the evaluation produced. What is on `results` is a
[`FunctionResults`][ropt.results.FunctionResults], the same kind a handler
receives, so everything is read the same way wherever it came from. Here
`result.variables` is just the point you supplied.

The runnable script is
[examples/evaluate.py](https://github.com/TNO-ropt/ropt/blob/main/examples/evaluate.py),
which evaluates a single vector and then a matrix of them.

## When something goes wrong

Not every problem is an exception. An optimization that cannot make progress
still returns normally, and indicates why in `result.exit_code` —
`TOO_FEW_REALIZATIONS` when not enough realizations produced a value, for
instance. An optimization that is aborted, for instance because another task
raised, raises [`AbortedError`][ropt.exceptions.AbortedError] instead (see
[Failures and Aborts](failures.md)).

[Exit Codes](../results/exit_codes.md) lists all exit codes, and explains why a
reason and a result are independent: `result.results` is `None` when no feasible
result was ever recorded, whatever the reason the optimization ended, and an
optimization that stops part-way still returns the best result it had reached
before that. The evaluations it did make reach the
[handlers](../results/handlers.md) attached to it either way.

What *is* raised falls into three groups:

- **Mistakes in the configuration** surface as a `pydantic.ValidationError`
  from the `config` dictionary: an unknown field, a value of the wrong type, a
  method name no installed plugin provides, or a set of options the chosen
  method does not accept. These are raised when the optimization starts, before
  anything is evaluated.

- **Mistakes in the call to `optimize` itself** can raise:
    - `ValueError` when an argument has the wrong shape, or the evaluation
      function returns the wrong number of values.
    - [`WorkflowError`][ropt.exceptions.WorkflowError] when an operation
      conflicts with what is already happening, such as starting another
      optimization from inside an evaluation or reusing an object that is
      already in use.
    - [`UnsupportedError`][ropt.exceptions.UnsupportedError] when an optional
      dependency is missing, or the chosen method cannot handle part of the
      configured problem.
    - [`ExecutionError`][ropt.exceptions.ExecutionError] when `ropt` cannot
      start or complete work needed for the optimization.
    - [`AbortedError`][ropt.exceptions.AbortedError] when the optimization or
      work it started is forcefully stopped before it finishes.

- **Exceptions from your own evaluation function** are not caught. They travel
  back from wherever the evaluation ran — including a worker thread or process —
  and are re-raised from the `optimize` call. Return `float("nan")` or
  `numpy.nan` instead if a failed realization should be tolerated rather than
  fatal.

The [`RoptError`][ropt.exceptions.RoptError] types above are imported from
`ropt` itself, and catching the base class catches all of them at once:

```python
from ropt import RoptError
```

It deliberately does not cover the first and third groups:
configuration errors belong to pydantic, and errors from your evaluation
function stay whatever you raised.

[Running Multiple Optimizations](many_runs.md) shows how several optimization
tasks can run in parallel. If a task raises an exception, every task started
from inside it is aborted, and the exception is raised in the code that started
the task. [Failures and Aborts](failures.md) describes each case.

## A note on enums

A few config values and result fields use enumerations.
[`ExitCode`][ropt.enums.ExitCode], the type of `result.exit_code`, is imported
from `ropt`:

```python
from ropt import ExitCode
```

The enumerations that name configuration values —
[`VariableType`][ropt.enums.VariableType],
[`BoundaryType`][ropt.enums.BoundaryType] and
[`PerturbationType`][ropt.enums.PerturbationType] — are in
[`ropt.enums`][ropt.enums]. Their members are strings, so a configuration
accepts either the member or the value it equals:

```python
from ropt.enums import VariableType

config = {"variables": {"variable_count": 2, "types": VariableType.INTEGER}}
```

Writing `"integer"` instead gives the same configuration. A validated
configuration reports these fields as enum members, whichever form was written.
