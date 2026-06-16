# Writing an Evaluator Callback

An evaluator is the bridge between `ropt` and your model. `ropt` decides which
variable vectors need values; the evaluator computes the objective (and
optional nonlinear-constraint) values for each.

There are two ways to provide an evaluator:

1. **A plain callable** — A callable adhering to the
   [`EvaluatorCallback`][ropt.evaluator.EvaluatorCallback] protocol is the
   simplest option, used by [`BasicOptimizer`][ropt.workflow.BasicOptimizer].
2. **An [`Evaluator`][ropt.workflow.evaluators.Evaluator] subclass** —
   required by the [workflow framework](workflows.md), and useful when you need
   state, caching, async execution, or HPC dispatch.

This page focuses on the usage of callables together with
[`BasicOptimizer`][ropt.workflow.BasicOptimizer] objects. For advanced usage of
evaluators within workflows, see [Optimization Workflows](workflows.md).

## The callable signature

Evaluator callbacks must adhere to the
[`EvaluatorCallback`][ropt.evaluator.EvaluatorCallback] protocol. For instance an
evaluator function should look like this:

```python
from numpy.typing import NDArray
import numpy as np
from ropt.evaluator import EvaluatorContext, EvaluatorResult


def my_evaluator(
    variables: NDArray[np.float64],
    context: EvaluatorContext,
) -> EvaluatorResult:
    ...
```

- `variables` has shape `(n_rows, n_variables)`. Each row is a separate
  variable vector to evaluate.
- `context` carries per-row metadata (see below) plus the immutable
  [`EnOptContext`][ropt.context.EnOptContext] for the run.
- The return value should be an
  [`EvaluatorResult`][ropt.evaluator.EvaluatorResult] object that packages
  objective values (and optional constraint values, evaluation info, and per-row
  error indicators).

## What's in `EvaluatorContext`

The [`EvaluatorContext`][ropt.evaluator.EvaluatorContext] dataclass exposes:

| Field           | Meaning
| --------------- | -------------------------------------------------------------------------------------------------------
| `context`       | The full [`EnOptContext`][ropt.context.EnOptContext] (read-only).
| `active`        | A boolean array indicating which rows actually need evaluation.
| `realizations`  | Integer realization index for each row.
| `perturbations` | Integer perturbation index per row, or `-1` for unperturbed rows. `None` if no perturbations are used.

Use `realizations` to pick the right per-realization model parameters (an
uncertainty draw, a different simulation deck, etc.). Use `active` to skip
rows that are not needed: utility methods
[`get_active_evaluations`][ropt.evaluator.EvaluatorContext.get_active_evaluations]
and
[`insert_inactive_results`][ropt.evaluator.EvaluatorContext.insert_inactive_results]
help filter the input and re-expand the output.

## Returning results

[`EvaluatorResult`][ropt.evaluator.EvaluatorResult] stores:

| Field              | Meaning
| ------------------ | --------------------------------------------------------------------------------------------------------------------
| `objectives`       | An array of shape `(n_rows, n_objectives)`.
| `constraints`      | An optional array of shape `(n_rows, n_nonlinear_constraints)`, required when `nonlinear_constraints` is configured.
| `evaluation_info`  | A dict of arrays carrying user metadata for each row. Not used internally by `ropt`; stored verbatim on results for application use (e.g., to link results back to input vectors).
| `batch_id`         | Optional integer identifying this set of evaluation results.

Inactive rows (where `active` is `False`) should have their result values set
to zero. Rows where an evaluation failed should be set to `np.nan` (see
[Handling partial failures](#handling-partial-failures) below).

## Returning constraints

```python
def evaluator(variables, context):
    obj = ...    # shape (n_rows, n_objectives)
    con = ...    # shape (n_rows, n_nonlinear_constraints)
    return EvaluatorResult(objectives=obj, constraints=con)
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
    return EvaluatorResult(objectives=obj)
```

Combined with the `realization_min_success` field of
[`RealizationsConfig`][ropt.config.RealizationsConfig], this allows the
optimization to continue as long as enough realizations succeed.

## Where to next

- Read the results: [Working with Results](results.md).
- Use evaluator subclasses for caching, async, or HPC dispatch:
  [Optimization Workflows](workflows.md).
