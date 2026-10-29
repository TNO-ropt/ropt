# Running Optimizations

!!! note
    This page calls [`optimize`][ropt.optimize] and [`evaluate`][ropt.evaluate]
    without a evaluation pool, so they evaluate one point at a time and return
    when they have finished. Parallel evaluation, several optimizations at once
    and aborting from outside need a pool from a [session](sessions.md).
    Everything this page explains applies there unchanged.

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

outcome = optimize(config, np.zeros(3), objective)
if outcome.results is not None:
    print(outcome.results.variables)         # the best variables found
    print(outcome.results.target_objective)  # the objective value there
```

The start point, the evaluation function's variable vector and per-variable
configuration arrays use the same positional order. You can add labels in the
optional `names` section of the config dictionary, but they do not change that
order; see
[Positions and labels](../getting_started/quickstart.md#positions-and-labels).

## Choose how to run

- To **optimize**, call [`optimize`][ropt.optimize] with the configuration, a
  start point and an evaluation function, as in the example above.
- To **evaluate without optimizing**, call [`evaluate`][ropt.evaluate] for one
  variable vector or [`evaluate_batch`][ropt.evaluate_batch] for several. See
  [Evaluating without optimizing](#evaluating-without-optimizing).
- To **run evaluations in parallel** or run several optimizations at once, use
  a pool from a [session](sessions.md). See
  [Evaluating in Parallel](parallel.md) and
  [Running Multiple Optimizations](multiple_runs.md).
- To **reuse a previous evaluation** at the optimization's start point, pass
  its function results as `f0`; pass gradient results as `g0` if available. See
  [Reusing Results at the Starting Point](initial_values.md).
- To **pass values to evaluations**, subscribe an event handler to
  `START_EVALUATION` and modify `event.metadata`. See
  [Passing values to evaluations from event handlers](#passing-values-to-evaluations-from-event-handlers).
- To **collect or inspect output**, use callbacks and result handlers. See
  [Getting Output](../results/output.md).

The [evaluation function](#the-evaluation-function) section explains what the
function receives and returns. The rest of this page covers results and
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
    - `context.batch_id` — a unique, increasing identifier for the evaluation
      batch. When multiple optimizations run in the same process, IDs may not
      start at 0 or be consecutive.
    - `context.row_idx` — the position of the evaluation within its batch,
      counted from 0.
    - `context.metadata` — the metadata for this batch, including any
      changes made by [`START_EVALUATION` handlers](#passing-values-to-evaluations-from-event-handlers),
      or an empty dictionary when none was supplied (see
      [Attaching metadata](#attaching-metadata)).

The function returns the objective value(s), followed by any constraint values.
There are three ways to return them:

- a **single number** when there is one objective and no nonlinear constraints;
- a **numpy vector** or a **sequence** of numbers when there are several
  objectives or nonlinear constraints — put the objectives first, then the
  constraints;
- an [`EvaluationFunctionResult`][ropt.EvaluationFunctionResult]
  when you also want to attach `metadata`; it holds `objectives`, `constraints`
  and `metadata` in separate fields.

With several objectives or nonlinear constraints, each value must match its
position in the corresponding configuration arrays, even if the entries have
labels in the config's `names` section.

If a realization fails to compute, return `float("nan")` or `numpy.nan` for it.
`ropt` treats `NaN` as a failed realization and keeps going, as long as enough
realizations succeed. How many is enough is set by
[`realization_min_success`](../optimizer_setup/configuration_sections.md#realizations),
which defaults to *all* of them — so a single `NaN` ends the optimization with
`TOO_FEW_REALIZATIONS` unless you lower it.

!!! warning
    Do not return `NaN` when an evaluation fails because of a Python exception,
    such as one caused by a programming or configuration error. Catching the
    exception and returning `NaN` would hide the error. Return `NaN` only when
    the function values cannot be computed, for example because a simulation
    failed.

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
outcome = optimize(config, x0, objective)

outcome.exit_code  # why the optimization stopped (an ExitCode)
outcome.results    # the best evaluation, or None if none was valid
```

`outcome.results` is a [`FunctionResults`][ropt.results.FunctionResults], so the best
point and its values are read from it at the paths described in [Working with
Results](../results/results.md):

```python
if outcome.results is not None:
    outcome.results.variables             # the best variables
    outcome.results.target_objective      # the objective value there
    outcome.results.functions.objectives  # the separate objective values
    outcome.results.functions.constraints # the nonlinear constraint values
```

`outcome.results` is `None` when the optimization produced no valid result: too few
realizations
succeeded, or no result ever satisfied the constraints — see
[When something goes wrong](#when-something-goes-wrong). One check therefore
covers every field.

## Reporting progress

Pass a result callback to track the optimization as it runs. It is called once
for every function evaluation, with the
[`FunctionResults`][ropt.results.FunctionResults] of that evaluation:

```python
def report(result):
    print(result.variables, result.target_objective)


optimize(config, x0, objective, result_callback=report)
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


optimize(config, x0, objective, result_callback=report, report_gradients=True)
```

A [`GradientResults`][ropt.results.GradientResults] records the ID of the
function evaluation whose values were used to construct it.
[`GradientResults.uses`][ropt.results.GradientResults.uses] checks whether a
[`FunctionResults`][ropt.results.FunctionResults] object has that ID; it does
not check whether the two results are at the same point. If point equality
matters, compare their `variables` values separately.

### Stopping early from the callback

The result callback doubles as a **user-defined stopping criterion**: return
`True` and the optimization stops gracefully after the current evaluation, with
exit code `STOPPED`. Any other return value (including `None`) lets it
continue.

```python
from ropt import ExitCode


def stop_if_good_enough(result):
    if result.target_objective is not None and result.target_objective < 1e-6:
        return True  # good enough — stop this optimization
    return None


outcome = optimize(config, x0, objective, result_callback=stop_if_good_enough)
assert outcome.exit_code is ExitCode.STOPPED
```

When [running multiple optimizations at once](multiple_runs.md), this stops only the
optimization whose callback returned `True`; the others continue. To abort all
optimizations running in the same session, call the session's
[`abort`][ropt.Session.abort] method. For its effect when several optimizations
run together, see [Aborting from outside](multiple_runs.md#stopping-from-outside).

The runnable script is
[examples/stopping.py](https://github.com/TNO-ropt/ropt/blob/main/examples/stopping.py),
which stops an optimization after a fixed number of results and then reads the
best point it had reached.

## Attaching metadata

You can attach arbitrary **metadata** to an optimization or evaluation, from two
sources:

- **Run defaults** — pass a `metadata` dict to `optimize`, `evaluate`, or
  `evaluate_batch`. `ropt` copies these values at run start and uses them as
  defaults for each batch, which is one way to tag an optimization:

  ```python
  outcome = optimize(config, x0, objective, metadata={"id": 7})
  print(outcome.results.metadata["id"])   # 7
  ```

  Each batch's values reach the evaluation function as `context.metadata`
  and are copied onto its results. An
  [event handler for `START_EVALUATION`](#passing-values-to-evaluations-from-event-handlers) can add or
  override values for that batch. Without updates, the values stay the same
  throughout the run. With several optimizations at once, a run identifier
  distinguishes them; see [Give each optimization an ID](multiple_runs.md).

- **Per evaluation** — return an
  [`EvaluationFunctionResult`][ropt.EvaluationFunctionResult]
  from the evaluation function with a `metadata` field. For a function
  evaluation, this value is stored per realization, next to the objective
  values:

  ```python
  from ropt import EvaluationFunctionResult

  def objective(variables, context):
      value = ...
      residual = variables - x0
      return EvaluationFunctionResult(objectives=value, metadata={"residual": residual})

  outcome = optimize(config, x0, objective)
  print(outcome.results.evaluations.metadata["residual"].shape)  # (3, 3)
  ```

  When the evaluation function is called for perturbed points to estimate a
  gradient, its metadata is stored per realization and perturbation on
  `outcome.gradient.evaluations.metadata`, if a gradient result is available.

  A key does not have to be set on every evaluation. Missing entries get
  `np.nan` for numeric values and `None` otherwise, so a numeric column with
  missing entries is widened to `float64` — numpy has no integer NaN. A key
  set on every evaluation keeps its natural dtype, and mixing strings and
  non-strings under one key raises a `ValueError`.

  Returning an array instead of a scalar adds a dimension for its elements,
  named after the metadata key. This is a
  [user-defined axis](../results/results.md#user-defined-axes). For function
  evaluations, the metadata array has dimensions for realizations and the
  metadata elements. For gradient evaluations, it also has a dimension for
  perturbations. The `names` section of the configuration can label the
  metadata elements, and the
  [`DataFrameHandler`](../results/handlers.md#dataframehandler) exports them
  as columns.

  In the example above, each row corresponds to a realization and each column
  to an element of `residual`. For metadata returned while evaluating
  perturbed points, there is also a dimension for perturbations. The
  [`DataFrameHandler`](../results/handlers.md#dataframehandler) example shows
  how to label and export the metadata elements.

Neither kind is interpreted by `ropt`. The effective batch metadata ends up on
`outcome.results.metadata`. Metadata returned for function evaluations is on
`outcome.results.evaluations.metadata` (one entry per realization); metadata
returned for perturbed evaluations is on
`outcome.gradient.evaluations.metadata` (one entry per realization and
perturbation). Both kinds can be tabulated as columns by the
[`DataFrameHandler`](../results/handlers.md#dataframehandler). See [Working with
Results](../results/results.md#metadata) for how each appears. The full runnable
script is
[examples/metadata.py](https://github.com/TNO-ropt/ropt/blob/main/examples/metadata.py).

## Passing values to evaluations from event handlers

An event handler subscribed to `START_EVALUATION` runs before each batch of
function or gradient evaluations. It can compute values and pass them to the
evaluation function by reading and modifying
[`event.metadata`][ropt.events.EnOptEvent].
Any required evaluation-function calls start after all handlers for that event
have returned.

The runnable script
[examples/callback_metadata.py](https://github.com/TNO-ropt/ropt/blob/main/examples/callback_metadata.py)
optimizes a noisy objective. Before each batch, it draws one noise value per
realization and passes them to the evaluations. The callback draws the noise
and adds it to `event.metadata`. It reads the number of realizations from
`event.context`:

```python
--8<-- "examples/callback_metadata.py:draw_noise"
```

`event.metadata` is a dictionary: handlers can read, add, replace, or delete
entries for this batch, or assign a new dictionary to `event.metadata`. Before
the next `START_EVALUATION`, the dictionary resets to the values supplied when
the run started. Handlers for the same event see each other's changes in the
order they run.

The evaluation function reads the noise for its realization from
`context.metadata`:

```python
--8<-- "examples/callback_metadata.py:objective"
```

Treat this dictionary as input; evaluations in the same process may share it.
Return per-realization information through
[`EvaluationFunctionResult`][ropt.EvaluationFunctionResult] instead, as
described under [Attaching metadata](#attaching-metadata).

To run the callback before each batch, subscribe a
[`CallbackHandler`][ropt.CallbackHandler] to `START_EVALUATION`:

```python
--8<-- "examples/callback_metadata.py:run"
```

All evaluations in a batch, including those at perturbed points, use the same
noise for a given realization. Each result records the noise of its batch in
`result.metadata["noise"]`.

The same handler works with [`evaluate`][ropt.evaluate] and
[`evaluate_batch`][ropt.evaluate_batch], and when using a pool. With workers in
other processes, metadata must be serializable. See the [event metadata
rules](../results/handlers.md#updating-evaluation-metadata) for ordering,
lifetime, and result reuse.

!!! note "Optimizations running at once"
    A handler shared by [optimizations running at once](multiple_runs.md) runs
    its callback for one of them at a time. The callback in the example is not
    thread-safe, because all calls use the same random number generator: share
    one handler between such optimizations, or give each optimization its own
    handler and generator. A thread-safe callback can be used by a separate
    handler for each optimization, so its calls can overlap; see
    [Updating evaluation metadata](../results/handlers.md#updating-evaluation-metadata).

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
still returns normally, and indicates why in `outcome.exit_code` —
`TOO_FEW_REALIZATIONS` when not enough realizations produced a value, for
instance. An optimization that is aborted, for instance because another
function given to the same `run` call raised, raises
[`AbortedError`][ropt.exceptions.AbortedError] instead (see
[Failures and Aborts](failures.md)).

[Exit Codes](../results/exit_codes.md) lists all exit codes, and explains why a
reason and a result are independent: `outcome.results` is `None` when no
feasible result was ever recorded, whatever the reason the optimization ended,
and an optimization that stops part-way still returns the best result it had
reached before that. The evaluations it did make reach the
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

[Running Multiple Optimizations](multiple_runs.md) shows how several optimizations
can run in parallel. If a call to `optimize`, `evaluate`, `evaluate_batch` or
`run` raises an exception, every call made from inside it is aborted, and the
exception is raised in the code that made the call.
[Failures and Aborts](failures.md) describes each case.

## A note on enums

A few config values and result fields use enumerations.
[`ExitCode`][ropt.enums.ExitCode], the type of `outcome.exit_code`, is imported
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
