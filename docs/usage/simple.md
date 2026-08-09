# The Simple API

The `ropt.simple` module is the easy way to run an optimization. You import
everything you need from one place:

```python
from ropt.simple import optimize
```

You give `optimize` three things:

- a **config** dictionary that describes the problem,
- a **start point** (the first set of variable values),
- an **objective function** that returns the value you want to make as small as
  possible.

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

That is the whole pattern. The rest of this page explains each part and the
extra things you can do.

## The objective function

Your objective is a plain Python function with two arguments:

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
succeed (see [Configuration](configuration.md)).

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
[Working with Results](results.md).

## Reporting progress

Pass a `report` callback to watch the optimization as it runs. It is called once
for every function evaluation, with an [`EvaluateResult`][ropt.simple.EvaluateResult]:

```python
def report(result):
    print(result.target_objective)


optimize(config, x0, objective, report=report)
```

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
[Running in Parallel](execution.md) for a full explanation of the three choices
and their trade-offs:

```python
from ropt.simple import processes

with processes(workers=8):
    result = optimize(config, x0, objective)
```

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

## Collecting results across runs

An [`optimize`][ropt.simple.optimize] call returns only the best result. To
collect *every* result — for example across a loop of runs — use a
[`handlers`][ropt.simple.handlers] block with a
[`HistoryHandler`][ropt.simple.HistoryHandler]:

```python
from ropt.simple import HistoryHandler, handlers, optimize

history = HistoryHandler()
with handlers(history):
    for x0 in start_points:
        optimize(config, x0, objective)

print(history.results)   # all results from every run in the block
```

The handlers ([`HistoryHandler`][ropt.simple.HistoryHandler],
[`ResultsHandler`][ropt.simple.ResultsHandler],
[`DataFrameHandler`][ropt.simple.DataFrameHandler]) are imported straight from
`ropt.simple`.

## A note on enums

A few config values and result fields use enumerations, such as
[`VariableType`][ropt.enums.VariableType] for integer variables and
[`ExitCode`][ropt.enums.ExitCode] for `result.exit_code`. These are **not** part
of `ropt.simple`; import them from [`ropt.enums`][ropt.enums]:

```python
from ropt.enums import ExitCode, VariableType
```
