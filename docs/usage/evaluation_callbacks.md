# Writing Evaluation Callbacks

During optimization `ropt` decides which variable vectors need values for the
objectives and optional nonlinear-constraints. The user must provide the code to
perform these function evaluations.

There are two ways to provide such evaluation code to
[`BasicOptimizer`][ropt.workflow.BasicOptimizer]:


1. **A plain callable** — A callable adhering to the
   [`EvaluationBatchCallback`][ropt.evaluation.EvaluationBatchCallback] protocol
   that receives all variable vectors at once as a 2-D array.
2. **An [`Evaluator`][ropt.workflow.evaluators.Evaluator] subclass** — Classes
    that support advanced features such as caching, parallel execution, or HPC
    dispatch. Here we only discuss
   [`FunctionEvaluator`][ropt.workflow.evaluators.FunctionEvaluator], a
   convenience wrapper around a simpler per-row function following the
   [`EvaluationFunctionCallback`][ropt.workflow.evaluators.EvaluationFunctionCallback] protocol.
   [`Evaluator`][ropt.workflow.evaluators.Evaluator] classes are discussed
   in more detail in
   [Optimization Workflows](workflows.md) and [Parallel Evaluation](parallel.md).

## The callable signature

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
| -------------- | --------------------------------------------------------------------------------------------------------------------
| `objectives`   | An array of shape `(n_rows, n_objectives)`.
| `constraints`  | An optional array of shape `(n_rows, n_nonlinear_constraints)`, required when `nonlinear_constraints` is configured.
| `metadata`     | A dict of arrays carrying user metadata for each row. Not used internally by `ropt`; stored verbatim on results for application use (e.g., to link results back to input vectors).
| `batch_id`     | Optional integer identifying this set of evaluation results.

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
the [`FunctionEvaluator`][ropt.workflow.evaluators.FunctionEvaluator] offers a
simpler alternative. Instead of receiving the full 2-D batch and managing the
loop yourself, you write a function that takes a single 1-D variable vector and
returns the objective (and optional constraint) values for that row. The
`FunctionEvaluator` handles the batching, the active-row filtering, and the
assembly of the final
[`EvaluationBatchResult`][ropt.evaluation.EvaluationBatchResult].

A function passed to `FunctionEvaluator` must follow the
[`EvaluationFunctionCallback`][ropt.workflow.evaluators.EvaluationFunctionCallback] protocol:

```python
from numpy.typing import NDArray
import numpy as np

from ropt.workflow.evaluators import (
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
  [`EvaluationFunctionContext`][ropt.workflow.evaluators.EvaluationFunctionContext]
  dataclass identifying the evaluation. It exposes:

    | Field          | Meaning
    | -------------- | -----------------------------------------------------------------------------------
    | `realization`  | Integer realization index for this row.
    | `perturbation` | Integer perturbation index, or `-1` when the evaluation is unperturbed.
    | `batch_id`     | Integer identifying the current evaluation batch.
    | `eval_idx`     | Row index within the batch.
    | `name`         | Optional task name set by the evaluator (e.g. via `AsyncEvaluator`'s `get_name` callback); `None` when no name was assigned.

- The return value is an
  [`EvaluationFunctionResult`][ropt.workflow.evaluators.EvaluationFunctionResult]
  dataclass with the following fields:

    | Field          | Meaning
    | -------------- | -----------------------------------------------------------------------------------
    | `objectives`   | The objective values as a scalar or 1-D array of length `n_objectives`.
    | `constraints`  | Optional constraint values as a scalar or 1-D array of length `n_nonlinear_constraints`.
    | `metadata`     | Optional `dict[str, Any]`; each entry is stored verbatim in the resulting [`EvaluationBatchResult.metadata`][ropt.evaluation.EvaluationBatchResult] for this row.

To use it with
[`BasicOptimizer`][ropt.workflow.BasicOptimizer], wrap the function in a
[`FunctionEvaluator`][ropt.workflow.evaluators.FunctionEvaluator] and pass it as
the evaluator argument:

```python
from ropt.workflow import BasicOptimizer
from ropt.workflow.evaluators import FunctionEvaluator

optimizer = BasicOptimizer(
    config,
    FunctionEvaluator(function=my_function),
)
optimizer.run(initial_values)
```

## Where to next

- Read the results: [Working with Results](results.md).
- Use evaluator subclasses for caching, async, or HPC dispatch:
  [Optimization Workflows](workflows.md).
- See it in action: [Ensemble-based Optimization](../tutorials/ensemble.md)
  (batch callback) and
  [Using FunctionEvaluator](../tutorials/function_evaluator.md)
  (per-evaluation callback).
