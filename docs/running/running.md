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
- an **evaluation function** that returns the objective value(s) to minimize,
  followed by any constraint values.

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

## The evaluation function

The evaluation function is a Python function with two arguments:

```python
from ropt.simple import EvaluationFunctionContext


def objective(variables: np.ndarray, context: EvaluationFunctionContext) -> float:
    ...
```

- `variables` is a 1-D NumPy array: one set of variable values to evaluate.
- `context` is an
  [`EvaluationFunctionContext`][ropt.components.evaluators.EvaluationFunctionContext]
  that tells you *which* evaluation this is:
    - `context.realization` — the realization number (see
      [Ensembles](#optimizing-over-an-ensemble) below); the field you need most
      often.
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
result.constraints       # the nonlinear constraint values, or None
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

### Stopping early from the callback

The `report` callback doubles as a **user-defined stopping criterion**: return
`True` and the optimization stops gracefully after the current evaluation, with
exit code `USER_ABORT`. Any other return value (including `None`) lets it
continue.

```python
from ropt.enums import ExitCode


def report(result):
    if result.target_objective is not None and result.target_objective < 1e-6:
        return True  # good enough — stop this optimization
    return None


result = optimize(config, x0, objective, report=report)
assert result.exit_code is ExitCode.USER_ABORT
```

With [`optimize_many`](parallel.md#many-optimizations-at-once) this stops only the run
whose callback returned `True`; the other runs continue.

!!! note "Nothing to stop on an evaluation"
    [`evaluate`][ropt.simple.evaluate] and
    [`evaluate_many`][ropt.simple.evaluate_many] take `report=` as well, but
    there the return value is **ignored**. An evaluation is a single batch with
    no optimizer loop to interrupt, so the callback reports and nothing more.
    This is the permanent contract, not a gap to be filled later.

## Attaching metadata

You can attach arbitrary **metadata** to a run, from two sources:

- **Constant, per run** — pass a `metadata` dict to `optimize`, `optimize_many`,
  `evaluate`, or `evaluate_many`. `ropt` copies it onto every result the run
  produces, which is handy for tagging a run:

  ```python
  result = optimize(config, x0, objective, metadata={"run_id": 7})
  print(result.results.metadata["run_id"])   # 7
  ```

  The same dict also reaches the evaluation function itself, as
  `context.metadata` — handy when the evaluation needs to know which run it is
  part of. With `optimize_many`, this is the natural way to tell runs apart:
  give one dict (shared by all runs) or a list with one dict per run; see
  [Give each run an ID](parallel.md#many-optimizations-at-once).

- **Per evaluation** — return an
  [`EvaluationFunctionResult`][ropt.components.evaluators.EvaluationFunctionResult]
  from the evaluation function with a `metadata` field. This value is stored per
  realization, next to the objective values:

  ```python
  from ropt.simple import EvaluationFunctionResult

  def objective(variables, context):
      value = ...
      return EvaluationFunctionResult(objectives=value, metadata={"seconds": 1.3})
  ```

Neither kind is interpreted by `ropt`. Constant metadata ends up on
`result.results.metadata`; per-evaluation metadata on
`result.results.evaluations.metadata` (one entry per realization). Both kinds can
be tabulated as columns by the [`DataFrameHandler`](handlers.md#dataframehandler). See
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

## When something goes wrong

Not every problem is an exception. An optimization that cannot make progress
still returns normally, and says why in `result.exit_code`:
`TOO_FEW_REALIZATIONS` when not enough realizations produced a value,
`EXECUTOR_STOPPED` when the pool it was evaluating on was closed under it. In
both cases the result fields are `None`, so check `exit_code` before using
them. A plain [`evaluate`][ropt.simple.evaluate] has no `exit_code`; there the
`None` fields are the only sign that nothing usable came back.

What *is* raised falls into three groups:

- **Mistakes in the configuration** surface as a `pydantic.ValidationError`
  from the `config` dictionary: an unknown field, a value of the wrong type, a
  method name no installed plugin provides, or a set of options the chosen
  method does not accept. These are raised at the start of the call, before
  anything is evaluated.

- **Mistakes in the call itself** raise a `ValueError` — a start point of the
  wrong shape, an evaluation function returning the wrong number of values —
  or one of the [`RoptError`][ropt.exceptions.RoptError] types:
  [`WorkflowError`][ropt.exceptions.WorkflowError] when a pool or handler is
  used in a way it cannot be (a closed pool, a handler already claimed by
  another run),
  [`UnsupportedError`][ropt.exceptions.UnsupportedError] when an optional
  dependency is missing, or when the chosen method cannot handle the problem
  as configured — a constraint it does not support, for instance, which is
  checked as the run starts — and
  [`ExecutionError`][ropt.exceptions.ExecutionError] when the machinery that
  runs the evaluations cannot start or breaks down.

- **Exceptions from your own evaluation function** are not caught. They travel
  back from wherever the evaluation ran — including a worker thread or process
  — and are re-raised from the `optimize` call. Return `float("nan")` instead
  if a failed realization should be tolerated rather than fatal.

Catching [`RoptError`][ropt.exceptions.RoptError] catches all of `ropt`'s own
errors at once. It deliberately does not cover the first and third groups:
configuration errors belong to pydantic, and errors from your evaluation
function stay whatever you raised.

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

## A note on enums

A few config values and result fields use enumerations, such as
[`VariableType`][ropt.enums.VariableType] for integer variables and
[`ExitCode`][ropt.enums.ExitCode] for `result.exit_code`. These are **not** part
of `ropt.simple`; import them from [`ropt.enums`][ropt.enums]:

```python
from ropt.enums import ExitCode, VariableType
```

## Where to next

- Run evaluations in parallel, or several optimizations at once:
  [Parallel Execution and Many Runs](parallel.md).
- Collect or react to every result, not just the best one:
  [Result Handlers](handlers.md).
