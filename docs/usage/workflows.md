# Optimization Workflows

[Basic Optimization](basic.md) uses
[`BasicOptimizer`][ropt.workflow.BasicOptimizer] to drive a single optimization
run with a single evaluator. For anything more elaborate — multiple optimizers
in sequence, nested optimizations, custom event handling, parallel/async
execution — drop down to the workflow framework.

The framework has four concepts:

| Concept                                                                     | Role                                                                                            |
| --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| [`ComputeStep`][ropt.workflow.compute_steps.ComputeStep]                    | An executable unit of work (run an optimizer, run a single ensemble evaluation, etc.).          |
| [`EventHandler`][ropt.workflow.event_handlers.EventHandler]                 | A reactive object that observes events emitted by a compute step.                               |
| [`Evaluator`][ropt.workflow.evaluators.Evaluator]                           | The object a compute step uses to actually evaluate the model.                                  |
| [`Server`][ropt.workflow.servers.Server]                                    | Dispatches evaluation tasks to threads, processes, or an HPC cluster.                           |

The first three are covered below. Servers are only relevant for asynchronous
and parallel execution and are discussed in
[Parallel Evaluation](parallel.md).

Compute steps emit [`EnOptEvent`][ropt.events.EnOptEvent] objects at key
points during execution — for instance when an evaluation starts or finishes.
The most important event is
[`FINISHED_EVALUATION`][ropt.enums.EnOptEventType.FINISHED_EVALUATION], which
carries the generated [`Results`][ropt.results.Results] objects. Event
handlers are attached to a step and receive its events, allowing them to
track, store, or react to results as they arrive.

### The EnOptEvent object

Each event is an [`EnOptEvent`][ropt.events.EnOptEvent] dataclass with three
fields:

| Field                      | Type                                                               | Description                                          |
| -------------------------- | ------------------------------------------------------------------ | ---------------------------------------------------- |
| `event_type`               | [`EnOptEventType`][ropt.enums.EnOptEventType]                      | Which lifecycle point triggered the event.           |
| `context`                  | [`EnOptContext`][ropt.context.EnOptContext]                         | The optimizer context active at the time of the event. |
| `results`                  | `tuple[`[`Results`][ropt.results.Results]`, ...]`                  | Result objects (empty tuple when no results apply).  |

### Event types

The [`EnOptEventType`][ropt.enums.EnOptEventType] enumeration defines the
following event types:

| Event type                    | When it fires                                                  |
| ----------------------------- | -------------------------------------------------------------- |
| `START_OPTIMIZER`             | Just before an optimization algorithm begins iterating.       |
| `FINISHED_OPTIMIZER`          | Immediately after the optimizer finishes (success or error).  |
| `START_EVALUATION`            | Before evaluating functions (or gradients).                   |
| `FINISHED_EVALUATION`         | After evaluation completes — carries `results`.               |
| `START_ENSEMBLE_EVALUATOR`    | Before an `EvaluationStep` compute step begins.               |
| `FINISHED_ENSEMBLE_EVALUATOR` | After an `EvaluationStep` compute step finishes.              |

Most event handlers only need to listen for `FINISHED_EVALUATION`; the other
types are useful for logging, progress bars, or custom lifecycle hooks.

## A workflow you can read end to end

```python
import numpy as np
from numpy.typing import NDArray

from ropt.context import EnOptContext
from ropt.workflow.compute_steps import OptimizationStep
from ropt.workflow.evaluators import FunctionEvaluator
from ropt.workflow.event_handlers import ResultHandler

# 1. Build the configuration.
CONFIG = {
    "variables": {"variable_count": 3, "perturbation_magnitudes": 1e-6},
    "realizations": {"weights": [1.0] * 5},
}

# 2. Define a per-realization evaluation function.
def my_function(variables: NDArray[np.float64], **kwargs) -> NDArray[np.float64]:
    return np.array([(variables - 1.0) @ (variables - 1.0)])

# 3. Construct an evaluator that calls a per-realization Python function.
evaluator = FunctionEvaluator(function=my_function)

# 4. Build the compute step.
step = OptimizationStep(evaluator=evaluator)

# 5. Attach event handlers.
result_handler = ResultHandler()  # remember the best
step.add_event_handler(result_handler)

# 6. Run the step.
step.run(
    variables=np.array([0.5, 0.7, 0.9]),
    context=EnOptContext.model_validate(CONFIG),
)

# 7. Read best results from the handlers.
print(f"Optimal variables: {result_handler['results'].evaluations.variables}")
```

This is a minimal example of optimizing a simple deterministic function. A full
runnable example for optimizing the Rosenbrock function with uncertain
parameters can be found here:
[examples/rosenbrock.py](https://github.com/TNO-ropt/ropt/blob/main/examples/rosenbrock.py).

## Compute steps

Two compute steps ship with `ropt`:

- [`OptimizationStep`][ropt.workflow.compute_steps.OptimizationStep] — runs
  an optimization algorithm.
- [`EvaluationStep`][ropt.workflow.compute_steps.EvaluationStep] — runs
  a single ensemble evaluation (no optimizer). For example, useful for evaluating an
  optimum on a different ensemble, or on a sub-set of realizations.

Both compute steps require an
[`EnOptContext`][ropt.context.EnOptContext] and a `variables` argument
passed to their `run(...)` method. For `OptimizationStep`, this is a
single 1-D variable vector (the starting point). For `EvaluationStep`,
it may be a single vector or a 2-D matrix where each row is a variable
vector to evaluate. An optional `metadata` dictionary can be attached; if
provided, it is included in the [`Results`][ropt.results.Results] objects
emitted via the `FINISHED_EVALUATION` event.

### Events emitted by OptimizationStep

[`OptimizationStep`][ropt.workflow.compute_steps.OptimizationStep]
executes an optimization algorithm based on the provided context. It
iteratively performs function and potentially gradient evaluations, yielding a
sequence of [`FunctionResults`][ropt.results.FunctionResults] and
[`GradientResults`][ropt.results.GradientResults] objects.

The following events are emitted during execution:

- [`START_OPTIMIZER`][ropt.enums.EnOptEventType.START_OPTIMIZER]:
  Emitted just before the optimization process begins.
- [`START_EVALUATION`][ropt.enums.EnOptEventType.START_EVALUATION]: Emitted
  immediately before a batch of function or perturbation evaluations is
  performed.
- [`FINISHED_EVALUATION`][ropt.enums.EnOptEventType.FINISHED_EVALUATION]: Emitted
  after an evaluation completes. The event's `results` field carries the
  generated [`Results`][ropt.results.Results] objects. Event handlers
  typically listen for this event to process or track optimization progress.
- [`FINISHED_OPTIMIZER`][ropt.enums.EnOptEventType.FINISHED_OPTIMIZER]:
  Emitted after the entire optimization process concludes (successfully,
  or due to termination conditions or errors).

### Events emitted by EvaluationStep

[`EvaluationStep`][ropt.workflow.compute_steps.EvaluationStep]
evaluates a batch of variable vectors. The `variables` argument can be a
single 1-D vector (treated as one row) or a 2-D matrix where each row is a
variable vector. The evaluator performs a function evaluation for the full
batch and produces a tuple of
[`FunctionResults`][ropt.results.FunctionResults] objects.

The following events are emitted during execution:

- [`START_ENSEMBLE_EVALUATOR`][ropt.enums.EnOptEventType.START_ENSEMBLE_EVALUATOR]:
  Emitted before the evaluation process begins.
- [`START_EVALUATION`][ropt.enums.EnOptEventType.START_EVALUATION]: Emitted
  just before the batch evaluation is performed.
- [`FINISHED_EVALUATION`][ropt.enums.EnOptEventType.FINISHED_EVALUATION]:
  Emitted after the evaluation completes. The event's `results` field
  carries the generated `FunctionResults` objects. Event handlers typically
  listen for this event.
- [`FINISHED_ENSEMBLE_EVALUATOR`][ropt.enums.EnOptEventType.FINISHED_ENSEMBLE_EVALUATOR]:
  Emitted after the entire compute step, including result emission, is
  finished.

### Exit codes

Both `run()` methods return an [`ExitCode`][ropt.enums.ExitCode] indicating
why the step finished:

| Exit code                    | Meaning                                                       |
| ---------------------------- | ------------------------------------------------------------- |
| `OPTIMIZER_FINISHED`         | The optimizer terminated normally.                            |
| `ENSEMBLE_EVALUATOR_FINISHED`| The evaluator step completed normally.                        |
| `TOO_FEW_REALIZATIONS`       | Too few realizations were evaluated successfully.             |
| `MAX_FUNCTIONS_REACHED`      | Maximum number of function evaluations was reached.           |
| `MAX_BATCHES_REACHED`        | Maximum number of evaluation batches was reached.             |
| `USER_ABORT`                 | The optimization was aborted by the user.                     |
| `ABORT_FROM_ERROR`           | Aborted due to an error handled elsewhere.                    |

## Event handlers

Event handlers are attached to a compute step via its `add_event_handler`
method. Once attached, the handler receives every event the step emits.

The framework ships four reusable handlers:

| Handler                                                                  | Purpose                                                                |
| ------------------------------------------------------------------------ | ---------------------------------------------------------------------- |
| [`ResultHandler`][ropt.workflow.event_handlers.ResultHandler]            | Keep the best (or last) result. Backs `BasicOptimizer.results`.        |
| [`HistoryHandler`][ropt.workflow.event_handlers.HistoryHandler]          | Keep every result.                                                     |
| [`CallbackHandler`][ropt.workflow.event_handlers.CallbackHandler]        | Forward selected event types to a user callback.                       |
| [`TableHandler`][ropt.workflow.event_handlers.TableHandler]              | Append rows to a structured table per result.                          |

Handlers expose their state through dictionary access (`handler[key]`). By
convention, `ResultHandler` and `HistoryHandler` both use the key `"results"` —
e.g. `result_handler["results"]` or `history_handler["results"]`. `TableHandler`
uses the table name as key — e.g. `table["functions"]`.

### ResultHandler

[`ResultHandler`][ropt.workflow.event_handlers.ResultHandler] listens for
[`FINISHED_EVALUATION`][ropt.enums.EnOptEventType.FINISHED_EVALUATION] events
emitted from within an optimization workflow. It processes the
[`Results`][ropt.results.Results] objects contained within these events and
selects a single [`FunctionResults`][ropt.results.FunctionResults] object to
retain based on defined criteria.

The criteria for selection are:

- **`what='best'` (default):** Tracks the result with the lowest weighted
  objective value encountered so far.
- **`what='last'`:** Tracks the most recently received valid result.

Optionally, results can be filtered based on constraint violations using the
`constraint_tolerance` parameter. If provided, any result violating
constraints beyond this tolerance is ignored.

Tracking logic (comparing 'best' or selecting 'last') operates on the
results in the optimizer's domain. However, the final selected result
that is made accessible via dictionary access (`result_handler["results"]`) is
transformed to the user's domain (when `domain="user"`, the default).

If the domain type is `"user"`, the result is converted from the optimizer
domain to the user domain before being stored.

### HistoryHandler

[`HistoryHandler`][ropt.workflow.event_handlers.HistoryHandler] listens for
[`FINISHED_EVALUATION`][ropt.enums.EnOptEventType.FINISHED_EVALUATION] events
emitted by compute steps from within an optimization workflow. It collects all
[`Results`][ropt.results.Results] objects contained within these events and
stores them sequentially in memory.

The accumulated results are stored as a tuple and can be accessed via dictionary
access using the key `"results"` (e.g., `history_handler["results"]`). Each time
new results are received from a valid source, they are appended to this tuple.
Initially, `history_handler["results"]` is `None`.

If the domain type is `"user"`, the results are converted from the optimizer
domain to the user domain before being stored.

### CallbackHandler

[`CallbackHandler`][ropt.workflow.event_handlers.CallbackHandler] listens for
events and forwards them to a callback function. It is constructed with a set of
`event_types` to respond to and a single `callback`. When an event with a
matching type arrives, the callback is called with the
[`EnOptEvent`][ropt.events.EnOptEvent].

### TableHandler

[`TableHandler`][ropt.workflow.event_handlers.TableHandler] tracks results and
stores them in pandas DataFrames.

#### Tables

Tables can be generated for
[`FunctionResults`][ropt.results.FunctionResults] and
[`GradientResults`][ropt.results.GradientResults] respectively. Tables are
added via the `add_table` method, which takes a name, a type (either
`"functions"` or `"gradients"`), a column specification and an optional domain
type. The column specification determines which fields of the results are
stored in the table and how they are named. The domain type determines
whether the results are transformed to the user domain before being stored
in the table.

Tables are accessed by their name via dictionary syntax, for example, as
`handler["evaluations"]`.

!!! warning

    Tables are generated on the fly from internal data when accessing them
    in this way. When multiple accesses are needed, it is more efficient to
    first store them in a variable.

#### Column specification

Columns are specified by providing a dictionary that maps field names to
column titles. The keys denote the names of the fields, using attribute
syntax. For instance a `functions.objectives` key indicates that the
result should contain a column with objective values that are found in the
`objectives` field of the `functions` field of the result. The values
corresponding to the keys are used to provide the column names.

For example, passing this dictionary via the `columns` argument generates a
table containing the batch id, the values of all calculated objectives and
the vector of variables:

```python
{
    "batch_id": "Batch",
    "functions.objectives": "Objective",
    "evaluations.variables": "Variables",
}
```

Some fields may result in multiple columns in the DataFrame if their values
are vectors or matrices. For example, `evaluations.variables` will generate
a separate column for each variable. The table specification above may
generate a pandas DataFrame looking something like this:

```
    Batch   Objective,0  Variables,v0  Variables,v1  Variables,v2
0       0  1.309826e+02      0.500000      0.900000      1.300000
1       0  4.362553e+12    120.900265     20.698539    -90.578972
...
```

Here, because the variables are vectors of length 2, there are two variable
columns generated. The corresponding column names consist of the column
title and the name of the variable vector, separated by a comma. Note that
the `functions.objectives` column also contains a comma followed by a 0
value. This is because `functions.objectives` is also a vector of
values, there just happens to be only one objective. Its index is used
instead of a name, because no name was provided in the configuration of the
optimization. Fields may even have matrix values, in which case the column
names may contain two item names or indices separated by commas.

!!! tip "Changing the column name separator"

    By default a comma is used to separate fields in the column names if
    needed. The `sep` input can be used to provide an alternative separator.

    You can exploit this by specifying a newline as the separator and
    display a nicely formatted table using the `tabulate` package:

    ```python
    from tabulate import tabulate

    print(tabulate(table["functions"], headers="keys", showindex=False))
    ```

    which will show something like this using multi-line headers:

    ```
      Batch         Objective    Variables    Variables     Variables
                            0           v0           v1            v2
    -------  ----------------  -----------  -----------  ------------
          0           130.983          0.5          0.9           1.3
    ...
    ```

#### Default tables

The `set_default_tables` method can be used to add a set of default tables:

- For function results it generates these tables:
    - `"functions"`: contains a set of values of the calculated functions.
    - `"evaluations"`: contains a set of values for all evaluations.
    - `"constraints"`: contains a set of values for all constraints.
- For gradient results it generates these tables:
    - `"gradients"`: contains a set of values of the calculated gradients.
    - `"perturbations"`: contains a set of values for all perturbations.

#### Adding columns and retrieving all tables

A single column can be added to an existing table after creation using
`add_column(table_name, field_name, title)`.

To retrieve all tables at once as a dictionary mapping names to DataFrames,
use `get_tables()`.

#### Callback functionality

The tables are updated anytime a result is processed. A callback can be
registered via `set_callback` to react each time the tables change. The
callback signature is:

```python
def my_callback(event: EnOptEvent) -> None:
    ...
```

It receives the [`EnOptEvent`][ropt.events.EnOptEvent] that caused the
tables to be updated.

## Evaluators

Compute steps take an [`Evaluator`][ropt.workflow.evaluators.Evaluator]
instance — *not* the plain callable accepted by `BasicOptimizer`. Three
synchronous evaluators are provided, plus an asynchronous one described in
the [next section](parallel.md):

| Evaluator                                                                      | Interface                                                                                                                     |
| ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| [`BatchEvaluator`][ropt.workflow.evaluators.BatchEvaluator]                    | Batch: `f(variables_2d, context)` → `EvaluationBatchResult`.                                                                        |
| [`FunctionEvaluator`][ropt.workflow.evaluators.FunctionEvaluator]              | Per-row: `f(variables_1d, realization=..., ...)` → array or dict.                                                             |
| [`CachedEvaluator`][ropt.workflow.evaluators.CachedEvaluator]                  | Wraps another evaluator, caching results by variable vector.                                                                  |
| [`AsyncEvaluator`][ropt.workflow.evaluators.AsyncEvaluator]                    | Parallel evaluation via a [`Server`][ropt.workflow.servers.Server] — see [Parallel Evaluation](parallel.md).                  |

[`BatchEvaluator`][ropt.workflow.evaluators.BatchEvaluator] defers to a callable
callback that receives the full 2-D variable matrix and an
[`EvaluationBatchContext`][ropt.evaluation.EvaluationBatchContext], and returns
an [`EvaluationBatchResult`][ropt.evaluation.EvaluationBatchResult]. Use this
when you need the full batch (e.g. vectorized computation, or an external
simulator that accepts all rows at once). The callback has the same signature as
the callable accepted by `BasicOptimizer`.

[`FunctionEvaluator`][ropt.workflow.evaluators.FunctionEvaluator] stores a
single function that returns a value for each objective and constraint. The
function is called once per row of the evaluation batch with the variable
vector and keyword arguments `realization`, `perturbation`, `batch_id`, and
`eval_idx`. The `perturbation` value is `-1` when the evaluation is not a
perturbation (i.e. the unperturbed function evaluation). It should return
either:

- A 1-D NumPy array of length *n_objectives + n_constraints* (objectives
  followed by constraints), or
- A dictionary with a `"result"` key containing that array; any additional
  keys are stored as `evaluation_info` entries in the returned
  [`EvaluationBatchResult`][ropt.evaluation.EvaluationBatchResult].

[`CachedEvaluator`][ropt.workflow.evaluators.CachedEvaluator] wraps another
evaluator with result caching. It retrieves previously computed function results
from `EventHandler` instances specified as `sources` — typically a
`HistoryHandler` or `ResultHandler`. For each variable vector and realization,
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

Sources can be managed dynamically with `add_sources()` and
`remove_sources()`.

The `eval_cached()` method is available for derived classes that need access to
which evaluations were cache hits — it returns both the
[`EvaluationBatchResult`][ropt.evaluation.EvaluationBatchResult] and a
dictionary mapping evaluation indices to their cached
[`FunctionResults`][ropt.results.FunctionResults].

The evaluators above run each function call sequentially in the current
thread. For parallel evaluation — whether via worker threads, separate
processes, or an HPC cluster — an asynchronous evaluator is needed. See
[Parallel Evaluation](parallel.md) for details on
[`AsyncEvaluator`][ropt.workflow.evaluators.AsyncEvaluator] and the available
servers.

!!! tip "Reusing objectives and constraints"

    When defining multiple objectives, you may need to reuse the same
    underlying computation. For example, a total objective could consist of
    the mean of the realizations plus their standard deviation. Rather than
    evaluating all realizations twice, compute them once and return the
    values for both objectives from a single evaluator call.

## Where to next

- [Parallel Evaluation](parallel.md) — run evaluations off-process
  or on a cluster.
- [Workflow Tutorial](../tutorials/rosenbrock_workflow.md) — step-by-step
  example building a workflow from scratch.
- Full example:
  [examples/rosenbrock.py](https://github.com/TNO-ropt/ropt/blob/main/examples/rosenbrock.py).
