# Running Optimizations

The `ropt.simple` module covers running an optimization. Everything you need is
imported from a single module:

```python
from ropt.simple import optimize
```

You give [`optimize`][ropt.simple.optimize] three things:

- a **config** dictionary that describes the problem — its variables,
  objectives, constraints and components, covered in
  [Optimizer Setup](../optimizer_setup/key_concepts.md),
- a **start point** (the first set of variable values),
- an **evaluation function** that returns the objective value(s) to minimize,
  followed by any constraint values.

```python
import numpy as np
from ropt.simple import optimize

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
from ropt.simple import EvaluationFunctionContext


def objective(variables: np.ndarray, context: EvaluationFunctionContext) -> float:
    ...
```

- `variables` is a 1-D NumPy array: one set of variable values to evaluate.
- `context` is an
  [`EvaluationFunctionContext`][ropt.simple.EvaluationFunctionContext]
  that identifies *which* evaluation this is:
    - `context.realization` — the realization number, for a problem with an
      ensemble of realizations; `optimize` then minimizes the weighted average
      objective over all of them. See
      [Ensemble-Based Optimization](../getting_started/ensemble.md). The field
      you need most often.
    - `context.metadata` — the `metadata` dict the run was started with, if
      any (see [Attaching metadata](#attaching-metadata)).
    - `context.batch_id`, `context.eval_idx`, `context.perturbation` — identify
      the evaluation batch, its row, and (for a gradient perturbation) which
      one; rarely needed directly.

The function returns the objective value(s), followed by any constraint values.
There are three ways to return them:

- a **single number** when there is one objective and no nonlinear constraints;
- a **list** of numbers when there are several objectives or nonlinear
  constraints — put the objectives first, then the constraints;
- an [`EvaluationFunctionResult`][ropt.simple.EvaluationFunctionResult]
  when you also want to attach `metadata`; it holds `objectives`, `constraints`
  and `metadata` in separate fields, so nothing has to be ordered.

If a realization fails to compute, return `float("nan")` for it. `ropt` treats
`NaN` as a failed realization and keeps going, as long as enough realizations
succeed. How many is enough is set by
[`realization_min_success`](../optimizer_setup/configuration_sections.md#realizations),
which defaults to *all* of them — so a single `NaN` ends the run with
`TOO_FEW_REALIZATIONS` unless you lower it.

## The result

`optimize` returns an [`OptimizationResult`][ropt.simple.OptimizationResult],
which carries two things:

```python
result = optimize(config, x0, objective)

result.exit_code  # why the run stopped (an ropt.enums.ExitCode)
result.results      # the best evaluation, or None if none was valid
```

`results` is a [`FunctionResults`][ropt.results.FunctionResults] — the same
object a [handler](../results/handlers.md) receives — so the best point and its
values are read from it at the paths described in
[Working with Results](../results/results.md):

```python
if result.results is not None:
    result.results.variables             # the best variables
    result.results.target_objective      # the objective value there
    result.results.functions.objectives  # the separate objective values
    result.results.functions.constraints # the nonlinear constraint values
```

`results` is `None` when the run produced no valid result: too few realizations
succeeded, or no result ever satisfied the constraints — see
[When something goes wrong](#when-something-goes-wrong). One check therefore
covers every field.

## Reporting progress

Pass a `report` callback to watch the optimization as it runs. It is called once
for every function evaluation, with the
[`FunctionResults`][ropt.results.FunctionResults] of that evaluation — the same
object a handler is given:

```python
def report(result):
    print(result.variables, result.target_objective)


optimize(config, x0, objective, report=report)
```

`result.variables` is the point that was evaluated. During an optimization the
optimizer chose it, so this is how you follow the path a run takes.

### Reporting gradients as well

Set `report_gradients=True` and the callback is also called with each
[`GradientResults`][ropt.results.GradientResults], which arrives in its own
evaluation. The callback then receives both kinds and has to tell them apart:

```python
from ropt.results import FunctionResults


def report(result):
    if isinstance(result, FunctionResults):
        print("objective", result.target_objective)
    else:
        print("gradient", result.target_gradient)


optimize(config, x0, objective, report=report, report_gradients=True)
```

A [`GradientResults`][ropt.results.GradientResults] carries a `function_key`
identifying the function evaluation it was computed from; the matching
[`FunctionResults`][ropt.results.FunctionResults] has the same value in its own
`function_key`.

### Stopping early from the callback

The `report` callback doubles as a **user-defined stopping criterion**: return
`True` and the optimization stops gracefully after the current evaluation, with
exit code `STOPPED`. Any other return value (including `None`) lets it
continue.

```python
from ropt.enums import ExitCode


def report(result):
    if result.target_objective is not None and result.target_objective < 1e-6:
        return True  # good enough — stop this optimization
    return None


result = optimize(config, x0, objective, report=report)
assert result.exit_code is ExitCode.STOPPED
```

With [`optimize_many`](many_runs.md) this stops only the run
whose callback returned `True`; the other runs continue.

The runnable script is
[examples/simple/stopping.py](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/stopping.py),
which stops a run after a fixed number of results and then reads the best point
it had reached.

!!! note "Nothing to stop on an evaluation"
    [`evaluate`][ropt.simple.evaluate] and
    [`evaluate_batch`][ropt.simple.evaluate_batch] take `report=` as well, but
    there the return value is **ignored**. An evaluation is a single batch with
    no optimizer loop to interrupt, so the callback reports and nothing more.

### Aborting a run from outside { #stopping-from-outside }

The `report` callback runs inside the run it stops, which is no use to a signal
handler or a user interface. [`Session.abort`][ropt.simple.Session.abort] is
the one that is called from another thread: it cuts off every run that belongs
to the session, and each ends with `USER_ABORT`, keeping the best result it had
reached.

```python
with session() as s:
    pool = s.thread_pool(workers=4)
    signal.signal(signal.SIGINT, lambda *_: s.abort())
    result = pool.optimize(config, x0, objective)

if result.exit_code is ExitCode.USER_ABORT:
    print("cut off early, best so far:", result.results)
```

The two are not the same kind of ending, which is why they have different exit
codes. A `report` callback stops the run *on a criterion*, at a point it chose.
An abort applies no criterion: the result is whatever the run had reached.

A run is cut off at its next evaluation boundary, so the evaluations already in
flight are still carried out and their workers are free only once they return.

`abort()` reaches the runs that are under way at the moment of the call, and
nothing more. A run started afterwards is unaffected, so a
loop that abandons one attempt and starts another keeps working. An
[`optimize_many`][ropt.simple.Session.optimize_many] counts as one run here: a
run it has queued behind its `limit` is cut off as well, and reports `USER_ABORT`
without evaluating anything. Leaving the session's `with` block aborts its runs
as well, and then releases its pools, which is what refuses a run started after
that. A run cut off that way reports `ABORTED` rather than `USER_ABORT`, since
it was the block ending rather than a request to stop.

A run started with the module-level [`optimize`][ropt.simple.optimize] belongs
to no session you hold and cannot be aborted this way. It evaluates on the
calling thread, which is the thread that would have to call `abort()`. Start it
on a session or one of its pools to make it abortable.

## Attaching metadata

You can attach arbitrary **metadata** to a run, from two sources:

- **Constant, per run** — pass a `metadata` dict to `optimize`, `optimize_many`,
  `evaluate`, or `evaluate_batch`. `ropt` copies it onto every result the run
  produces, which is one way to tag a run:

  ```python
  result = optimize(config, x0, objective, metadata={"run_id": 7})
  print(result.results.metadata["run_id"])   # 7
  ```

  The same dict also reaches the evaluation function itself, as
  `context.metadata` — useful when the evaluation needs to know which run it is
  part of. With `optimize_many`, this is how runs are told apart:
  give one dict (shared by all runs) or a list with one dict per run; see
  [Give each run an ID](many_runs.md).

- **Per evaluation** — return an
  [`EvaluationFunctionResult`][ropt.simple.EvaluationFunctionResult]
  from the evaluation function with a `metadata` field. This value is stored per
  realization, next to the objective values:

  ```python
  from ropt.simple import EvaluationFunctionResult

  def objective(variables, context):
      value = ...
      return EvaluationFunctionResult(objectives=value, metadata={"seconds": 1.3})
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

Neither kind is interpreted by `ropt`. Constant metadata ends up on
`result.results.metadata`; per-evaluation metadata on
`result.results.evaluations.metadata` (one entry per realization). Both kinds can
be tabulated as columns by the
[`DataFrameHandler`](../results/handlers.md#dataframehandler). See
[Working with Results](../results/results.md#metadata) for how
each appears in the pandas export. The full runnable script is
[examples/simple/metadata.py](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/metadata.py).

## Evaluating without optimizing

Sometimes you only want the objective value for a point, without running an
optimizer. Use [`evaluate`][ropt.simple.evaluate] for one point and
[`evaluate_batch`][ropt.simple.evaluate_batch] for several:

```python
from ropt.simple import evaluate, evaluate_batch

single = evaluate(config, x, objective)
batch = evaluate_batch(config, matrix, objective)

single.results  # one FunctionResults, or None if the evaluation was cut off
batch.results   # one per row of the matrix, empty if the batch was cut off
```

Both return an [`EvaluationResult`][ropt.simple.EvaluationResult], shaped like
the [`OptimizationResult`][ropt.simple.OptimizationResult] that `optimize`
returns: `exit_code` indicates why the evaluation ended and `results` holds what it
produced. What is on `results` is a
[`FunctionResults`][ropt.results.FunctionResults], the same kind a handler
receives, so everything is read the same way wherever it came from. Here
`result.variables` is just the point you supplied; it is only informative in a
`report` callback, where the optimizer chose the point.

An evaluation is a single batch, so it produces either every result or none.
That is why `exit_code` can only be `FINISHED`, or `USER_ABORT` and
`ABORTED_ON_ERROR` when [`Session.abort`](#stopping-from-outside) or a failing
run on the same session cut it off. An abort that arrives after the batch has
finished leaves it unaffected, and the evaluation reports `FINISHED`.

The runnable script is
[examples/simple/evaluate.py](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/evaluate.py),
which evaluates a single vector and then a matrix of them.

## When something goes wrong

Not every problem is an exception. An optimization that cannot make progress
still returns normally, and indicates why in `result.exit_code` —
`TOO_FEW_REALIZATIONS` when not enough realizations produced a value, for
instance, or `ABORTED_ON_ERROR` when another run on the same session raised and
brought this one down with it (see
[Failure in one run](many_runs.md#failure-in-one-run)).
[Exit Codes](../results/exit_codes.md) lists them all, and explains why a
reason and a result are independent: `result.results` is `None` when no feasible
result was ever recorded, whatever the reason the run ended, and a run that fails
part-way still returns the best result it had reached before that. The
evaluations it did make reach the [handlers](../results/handlers.md) attached to
it either way.

What *is* raised falls into three groups:

- **Mistakes in the configuration** surface as a `pydantic.ValidationError`
  from the `config` dictionary: an unknown field, a value of the wrong type, a
  method name no installed plugin provides, or a set of options the chosen
  method does not accept. These are raised at the start of the call, before
  anything is evaluated.

- **Mistakes in the call itself** raise a `ValueError` — a start point of the
  wrong shape, an evaluation function returning the wrong number of values —
  or one of the [`RoptError`][ropt.exceptions.RoptError] types:
  [`WorkflowError`][ropt.exceptions.WorkflowError] when a pool or handler
  is used in a way it cannot be (a pool asked to run work from one of its
  own workers, a pool whose session has closed, a handler already claimed by
  another run),
  [`UnsupportedError`][ropt.exceptions.UnsupportedError] when an optional
  dependency is missing, or when the chosen method cannot handle the problem
  as configured — a constraint it does not support, for instance, which is
  checked as the run starts — and
  [`ExecutionError`][ropt.exceptions.ExecutionError] when the worker pool
  that runs your evaluations, or a call handed to
  [`WorkerPool.offload`][ropt.simple.WorkerPool.offload], cannot start or breaks
  down, and [`AbortedError`][ropt.exceptions.AbortedError] when an abort
  abandoned an offloaded call.

- **Exceptions from your own evaluation function** are not caught. They travel
  back from wherever the evaluation ran — including a worker thread or process
  — and are re-raised from the `optimize` call. Return `float("nan")` instead
  if a failed realization should be tolerated rather than fatal. With
  [`optimize_many`][ropt.simple.optimize_many] there is no single exception to
  re-raise, so the call raises
  [`RunsFailedError`][ropt.exceptions.RunsFailedError] carrying what every run
  raised or reached; see
  [Failure in one run](many_runs.md#failure-in-one-run).

Catching [`RoptError`][ropt.exceptions.RoptError] catches all of `ropt`'s own
errors at once. It deliberately does not cover the first and third groups:
configuration errors belong to pydantic, and errors from your evaluation
function stay whatever you raised.

## A note on enums

A few config values and result fields use enumerations, such as
[`VariableType`][ropt.enums.VariableType] for integer variables and
[`ExitCode`][ropt.enums.ExitCode] for `result.exit_code`. These are
**not** part of `ropt.simple`; import them from [`ropt.enums`][ropt.enums]:

```python
from ropt.enums import ExitCode, VariableType
```
