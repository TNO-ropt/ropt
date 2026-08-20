# Writing Evaluation Callbacks

During optimization `ropt` decides which variable vectors need values for the
objectives and optional nonlinear-constraints. A compute step does not compute
these values itself — it delegates to an
[`Evaluator`][ropt.components.evaluators.Evaluator] instance that you supply. This
page describes the evaluators `ropt` provides and how to write the evaluation
code they wrap.

## The evaluators

| Evaluator                                                                      | Interface                                                                                                                     |
| ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| [`BatchEvaluator`][ropt.components.evaluators.BatchEvaluator]                    | Batch: `f(variables_2d, context)` → `EvaluationBatchResult`.                                                                  |
| [`FunctionEvaluator`][ropt.components.evaluators.FunctionEvaluator]              | Per-row: `f(variables_1d, context)` → `EvaluationFunctionResult`.                                                             |
| [`CachedEvaluator`][ropt.components.evaluators.CachedEvaluator]                  | Wraps another evaluator, caching results by variable vector.                                                                  |
| [`ParallelEvaluator`][ropt.components.evaluators.ParallelEvaluator]              | Parallel evaluation via an [`Executor`][ropt.components.executors.Executor] — see [Parallel Evaluation](parallel.md).           |

The first three run synchronously in the calling thread;
[`ParallelEvaluator`][ropt.components.evaluators.ParallelEvaluator] dispatches
work to an [`Executor`][ropt.components.executors.Executor] and is described in
[Parallel Evaluation](parallel.md).

!!! warning "Evaluators are not safe for concurrent use"

    An evaluator raises a
    [`WorkflowError`][ropt.exceptions.WorkflowError] if two threads execute its
    `eval` method at the same time. Serial reuse is allowed: a single evaluator
    instance may be shared by several compute steps that run one after another,
    even on different threads (for example, reusing one `FunctionEvaluator`
    across nested inner optimizations to keep batch ids counting). Do **not**
    share one evaluator across steps that run in parallel; give each parallel
    step its own evaluator. For the constraints on where each layer of a nested
    workflow may run, see
    [Nested workflows and process boundaries](parallel.md#nested-workflows-and-process-boundaries).
    Note that the parallelism of
    [`ParallelEvaluator`][ropt.components.evaluators.ParallelEvaluator] happens
    *below* `eval` — it dispatches tasks to an executor, so its own `eval` is
    still called on a single thread. An evaluator cannot be transferred to
    another process; serializing one reconstructs it as an inert placeholder in
    the worker, and `ropt` raises a
    [`TransferError`][ropt.exceptions.TransferError] before the task runs.

You write evaluation code for two of them.
[`BatchEvaluator`][ropt.components.evaluators.BatchEvaluator] takes a callback
that receives the full 2-D batch of variable vectors;
[`FunctionEvaluator`][ropt.components.evaluators.FunctionEvaluator] wraps a
simpler function called once per row. Both are covered below, followed by
[`CachedEvaluator`][ropt.components.evaluators.CachedEvaluator], which wraps
another evaluator to reuse previously computed results.

## Writing a batch callback

Evaluation callbacks must adhere to the
[`EvaluationBatchCallback`][ropt.evaluation.EvaluationBatchCallback] protocol.
For instance an evaluator function should look like this:

```python
from numpy.typing import NDArray
import numpy as np
from ropt.evaluation import EvaluationBatchContext, EvaluationBatchResult


def my_evaluator(
    variables: NDArray[np.float64],
    context: EvaluationBatchContext,
) -> EvaluationBatchResult:
    ...
```

- `variables` has shape `(n_rows, n_variables)`. Each row is a separate
  variable vector to evaluate.
- `context` carries per-row metadata (see below) plus the immutable
  [`EnOptContext`][ropt.context.EnOptContext] for the run.
- The return value should be an
  [`EvaluationBatchResult`][ropt.evaluation.EvaluationBatchResult] object that
  packages objective values (and optional constraint values, metadata,
  and per-row error indicators).

One advantage of this approach is that the callback receives all variable
vectors at once as a 2-D NumPy array. This makes it possible to exploit NumPy's
vectorized operations to evaluate all rows in a single pass, avoiding explicit
Python loops and achieving better performance.

## What is in `EvaluationBatchContext`

The [`EvaluationBatchContext`][ropt.evaluation.EvaluationBatchContext] dataclass exposes:

| Field           | Meaning
| --------------- | -------------------------------------------------------------------------------------------------------
| `context`       | The full [`EnOptContext`][ropt.context.EnOptContext] (read-only).
| `active`        | A boolean array indicating which rows actually need evaluation.
| `realizations`  | Integer realization index for each row.
| `perturbations` | Integer perturbation index per row, or `-1` for unperturbed rows. `None` if no perturbations are used.

Use `realizations` to pick the right per-realization model parameters (an
uncertainty draw, a different simulation deck, etc.). Use `active` to skip rows
that are not needed: utility methods
[`get_active_evaluations`][ropt.evaluation.EvaluationBatchContext.get_active_evaluations]
and
[`insert_inactive_results`][ropt.evaluation.EvaluationBatchContext.insert_inactive_results]
help filter the input and re-expand the output.

## Returning results

[`EvaluationBatchResult`][ropt.evaluation.EvaluationBatchResult] stores:

| Field          | Meaning
| -------------- | ----------------------------------------------------------
| `objectives`   | Objective values, shape `(n_rows, n_objectives)`.
| `constraints`  | Optional constraint values, shape `(n_rows, n_nonlinear_constraints)`.
| `metadata`     | Optional per-row metadata dict; not used by `ropt`.
| `batch_id`     | Batch label (default `0`).

`constraints` is required when `nonlinear_constraints` is configured in the
problem. `metadata` is stored verbatim on the resulting
[`Results`][ropt.results.Results] object and is useful for linking results back
to the input vectors that produced them. `batch_id` defaults to `0`; all
results will carry this label unless you set it yourself. For
auto-incrementing IDs pass a
[`BatchIdCounter`][ropt.components.evaluators.BatchIdCounter] (or any
`Callable[[], int]`) to the `batch_id_callback` argument of
[`FunctionEvaluator`][ropt.components.evaluators.FunctionEvaluator] or
[`ParallelEvaluator`][ropt.components.evaluators.ParallelEvaluator]; for raw
[`BatchEvaluator`][ropt.components.evaluators.BatchEvaluator] callbacks set it
yourself.

Inactive rows (where `active` is `False`) should have their result values set
to zero. Rows where an evaluation failed should be set to `np.nan` (see
[Handling partial failures](#handling-partial-failures) below).

## Returning constraints

```python
def evaluator(variables, context):
    obj = ...    # shape (n_rows, n_objectives)
    con = ...    # shape (n_rows, n_nonlinear_constraints)
    return EvaluationBatchResult(objectives=obj, constraints=con)
```
The constraint values are compared to the `lower_bounds` / `upper_bounds`
declared in
[`NonlinearConstraintsConfig`][ropt.config.NonlinearConstraintsConfig].

## Handling partial failures

If your evaluator cannot compute a given objective, set the corresponding entry
in the `objectives` field to `np.nan`. `ropt` treats NaN rows as failed
evaluations; the [`realization_min_success`][ropt.config.RealizationsConfig] and
[`perturbation_min_success`][ropt.config.GradientConfig] settings determine
whether the optimization can recover. For example:

```python
def evaluator(variables, context):
    n_rows, n_obj = variables.shape[0], 1
    obj = np.full((n_rows, n_obj), np.nan)
    for row in range(n_rows):
        try:
            obj[row, 0] = simulate(variables[row])
        except SimulationError:
            pass  # leave NaN
    return EvaluationBatchResult(objectives=obj)
```

Combined with the `realization_min_success` field of
[`RealizationsConfig`][ropt.config.RealizationsConfig], this allows the
optimization to continue as long as enough realizations succeed.

## Using `FunctionEvaluator`

When your evaluation function naturally works on a single variable vector at a
time — for instance when it calls an external simulator once per realization —
the [`FunctionEvaluator`][ropt.components.evaluators.FunctionEvaluator] offers a
simpler alternative. Instead of receiving the full 2-D batch and managing the
loop yourself, you write a function that takes a single 1-D variable vector and
returns the objective (and optional constraint) values for that row. The
`FunctionEvaluator` handles the batching, the active-row filtering, and the
assembly of the final
[`EvaluationBatchResult`][ropt.evaluation.EvaluationBatchResult].

A function passed to `FunctionEvaluator` must follow the
[`EvaluationFunctionCallback`][ropt.components.evaluators.EvaluationFunctionCallback] protocol:

```python
from numpy.typing import NDArray
import numpy as np

from ropt.components.evaluators import (
    EvaluationFunctionContext,
    EvaluationFunctionResult,
)


def my_function(
    variables: NDArray[np.float64],
    context: EvaluationFunctionContext,
) -> EvaluationFunctionResult:
    ...
```

- `variables` is a 1-D array for a single evaluation row.
- `context` is an
  [`EvaluationFunctionContext`][ropt.components.evaluators.EvaluationFunctionContext]
  dataclass identifying the evaluation. It exposes:

    | Field          | Meaning
    | -------------- | ----------------------------------------------------------
    | `realization`  | Integer realization index for this row.
    | `perturbation` | Perturbation index, or `-1` when unperturbed.
    | `batch_id`     | Integer identifying the current evaluation batch.
    | `eval_idx`     | Row index within the batch.

- The return value is an
  [`EvaluationFunctionResult`][ropt.components.evaluators.EvaluationFunctionResult]
  dataclass with the following fields:

    | Field          | Meaning
    | -------------- | ----------------------------------------------------------
    | `objectives`   | Scalar or 1-D array of length `n_objectives`.
    | `constraints`  | Optional scalar or 1-D array of length `n_nonlinear_constraints`.
    | `metadata`     | Optional `dict[str, Any]`; stored verbatim in the batch result.

    Each `metadata` entry is forwarded into
    [`EvaluationBatchResult.metadata`][ropt.evaluation.EvaluationBatchResult]
    for the corresponding row.

Wrap the function in a
[`FunctionEvaluator`][ropt.components.evaluators.FunctionEvaluator] and give the
evaluator to a compute step (see [Optimization Workflows](workflows.md)):

```python
from ropt.components.evaluators import FunctionEvaluator

evaluator = FunctionEvaluator(function=my_function)
```

## Using `CachedEvaluator`

[`CachedEvaluator`][ropt.components.evaluators.CachedEvaluator] wraps another
evaluator with result caching. It retrieves previously computed function results
from `EventHandler` instances specified as `sources` — typically a
`HistoryHandler` or `ResultsHandler`. For each variable vector and realization,
if a matching cached result is found, the cached objectives and constraints are
reused without calling the wrapped evaluator. Only uncached evaluations are
forwarded to the underlying evaluator.

Cache matching works as follows: for each requested variable vector and
realization, the evaluator searches through the `"results"` stored by its
sources. A match is found when the variables are equal (within floating-point
tolerance) and the realization matches. If realization names are configured,
they are used for matching (allowing cache hits across different optimization
runs with the same realization names). Otherwise, realization indices are used.

If some but not all evaluations are found in cache, the cached ones are
marked as inactive and only the missing evaluations are delegated to the
wrapped evaluator. The final combined result contains both cached and newly
computed values.

Sources can be added dynamically with `add_sources()`.

To record which evaluations were served from cache, pass a `hits_key` string
at construction time. When set, the returned
[`EvaluationBatchResult`][ropt.evaluation.EvaluationBatchResult] will contain
a boolean NumPy array in its `metadata` dictionary under that key —
`True` for evaluations that came from the cache, `False` for those that were
freshly computed.

The `eval_cached()` method is available for derived classes that need access to
which evaluations were cache hits — it returns both the
[`EvaluationBatchResult`][ropt.evaluation.EvaluationBatchResult] and a
dictionary mapping evaluation indices to their cached
[`FunctionResults`][ropt.results.FunctionResults].

## Using `ParallelEvaluator`

The evaluators above run each function call sequentially in the current thread.
For parallel evaluation — whether via worker threads, separate processes, or an
HPC cluster — use
[`ParallelEvaluator`][ropt.components.evaluators.ParallelEvaluator]. See
[Parallel Evaluation](parallel.md) for the evaluator and the available
executors.

!!! tip "Reusing objectives and constraints"

    When defining multiple objectives, you may need to reuse the same
    underlying computation. For example, a total objective could consist of
    the mean of the realizations plus their standard deviation. Rather than
    evaluating all realizations twice, compute them once and return the
    values for both objectives from a single evaluator call.

## Where to next

- Read the results: [Working with Results](../optimizer_setup/results.md).
- Run evaluations in parallel, in processes, or on a cluster:
  [Parallel Evaluation](parallel.md).
- See it in action: [Building a Workflow](../tutorials/workflow.md).
