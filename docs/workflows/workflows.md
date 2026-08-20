# Optimization Workflows

!!! note

    This is one of two ways to **run** an optimization; the other is
    [Running Optimizations](../running/running.md). What the optimization does —
    its variables, objectives, constraints, and components — is set up in
    [Optimizer Setup](../optimizer_setup/key_concepts.md), the same whichever way you run
    it.

The [simple API](../running/running.md) covers the common case with a single
call: one optimization run, one evaluator, results returned when it finishes. The
workflow components documented here are the layer beneath it — the same building
blocks the simple API is assembled from, exposed directly.

**When to use them.** Reach for the workflow components when the single-run model
of the simple API is too rigid: to chain several optimizers, nest optimizations
inside one another, react to results as they arrive through custom event
handlers, or drive evaluations across threads, processes, or an HPC cluster.
Wiring a workflow together by hand takes more code, but that code is where the
extra flexibility lives — these components can express applications the simple
API cannot.

**Who this is for.** This is the low-level API, and it assumes you are
comfortable with `asyncio` and threads. Parallel execution runs on an event
loop, event handlers may be invoked from several threads at once, and you are
responsible for respecting the concurrency and process-boundary rules spelled out
on these pages. Those rules are real limitations, not incidental detail: ignore
them and a workflow will raise rather than silently misbehave. In return you get
control that the simple API deliberately hides.

There are four core workflow components:

| Concept                                                                     | Role                                                                                            |
| --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| [`ComputeStep`][ropt.components.compute_steps.ComputeStep]                    | An executable unit of work (run an optimizer, run a single ensemble evaluation, etc.).          |
| [`EventHandler`][ropt.components.event_handlers.EventHandler]                 | A reactive object that observes events emitted by a compute step.                               |
| [`Evaluator`][ropt.components.evaluators.Evaluator]                           | The object a compute step uses to actually evaluate the model.                                  |
| [`Executor`][ropt.components.executors.Executor]                              | Dispatches evaluation tasks to threads, processes, or an HPC cluster.                           |

The first three are covered below. Executors are only relevant for asynchronous
and parallel execution and are discussed in [Parallel Evaluation](parallel.md).

Compute steps emit [`EnOptEvent`][ropt.events.EnOptEvent] objects at key
points during execution — for instance when an evaluation starts or finishes.
The most important event is
[`FINISHED_EVALUATION`][ropt.enums.EnOptEventType.FINISHED_EVALUATION], which
carries the generated [`Results`][ropt.results.Results] objects. Event
handlers are attached to a step and receive its events, allowing them to
track, store, or react to results as they arrive.

### The EnOptEvent object

Each event is an [`EnOptEvent`][ropt.events.EnOptEvent] dataclass with four
fields:

| Field                      | Type                                                               | Description                                          |
| -------------------------- | ------------------------------------------------------------------ | ---------------------------------------------------- |
| `event_type`               | [`EnOptEventType`][ropt.enums.EnOptEventType]                      | Which lifecycle point triggered the event.           |
| `context`                  | [`EnOptContext`][ropt.context.EnOptContext]                         | The optimizer context active at the time of the event. |
| `results`                  | `tuple[`[`Results`][ropt.results.Results]`, ...]`                  | Result objects (empty tuple when no results apply).  |
| `source`                   | [`ComputeStep`][ropt.components.compute_steps.ComputeStep]` \| None` | The compute step that emitted the event; call its `stop()` to stop that run. |

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
from ropt.components.compute_steps import OptimizationStep
from ropt.components.evaluators import (
    EvaluationFunctionContext,
    EvaluationFunctionResult,
    FunctionEvaluator,
)
from ropt.components.event_handlers import ResultsHandler

# 1. Build the configuration.
CONFIG = {
    "variables": {"variable_count": 3, "perturbation_magnitudes": 1e-6},
    "realizations": {"weights": [1.0] * 5},
}

# 2. Define a per-realization evaluation function.
def my_function(
    variables: NDArray[np.float64],
    context: EvaluationFunctionContext,
) -> EvaluationFunctionResult:
    return EvaluationFunctionResult(
        objectives=np.array([(variables - 1.0) @ (variables - 1.0)]),
    )

# 3. Construct an evaluator that calls a per-realization Python function.
evaluator = FunctionEvaluator(function=my_function)

# 4. Build the compute step.
step = OptimizationStep(evaluator=evaluator)

# 5. Attach event handlers.
result_handler = ResultsHandler()  # remember the best
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
runnable example that assembles the workflow components by hand can be found
here:
[examples/advanced/workflow.py](https://github.com/TNO-ropt/ropt/blob/main/examples/advanced/workflow.py).

## Compute steps

Two compute steps ship with `ropt`:

- [`OptimizationStep`][ropt.components.compute_steps.OptimizationStep] — runs
  an optimization algorithm.
- [`EvaluationStep`][ropt.components.compute_steps.EvaluationStep] — runs
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

[`OptimizationStep`][ropt.components.compute_steps.OptimizationStep]
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

[`EvaluationStep`][ropt.components.compute_steps.EvaluationStep]
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

The [`OptimizationStep`][ropt.components.compute_steps.OptimizationStep]'s
`run()` method returns an [`ExitCode`][ropt.enums.ExitCode] indicating why the
optimizer finished; the
[`EvaluationStep`][ropt.components.compute_steps.EvaluationStep]'s `run()`
returns nothing:

| Exit code                    | Meaning                                                       |
| ---------------------------- | ------------------------------------------------------------- |
| `OPTIMIZER_FINISHED`         | The optimizer terminated normally.                            |
| `TOO_FEW_REALIZATIONS`       | Too few realizations were evaluated successfully.             |
| `MAX_FUNCTIONS_REACHED`      | Maximum number of function evaluations was reached.           |
| `MAX_BATCHES_REACHED`        | Maximum number of evaluation batches was reached.             |
| `USER_ABORT`                 | An event handler requested a stop via `event.source.stop()`.  |
| `EXECUTOR_STOPPED`           | Aborted because the executor stopped before finishing.        |

An event handler can stop its own optimization by calling `event.source.stop()`
— for example after inspecting the `results` of a `FINISHED_EVALUATION` event and
deciding no further evaluations are worthwhile. The remaining handlers for that
event still run, and the optimizer then stops with `USER_ABORT` before the next
evaluation. Only the run that owns the emitting step is affected, so concurrent
optimizations continue. `stop()` merely sets a thread-safe flag, so it is safe
to call from a handler running behind an
[`EventDispatcher`][ropt.components.event_handlers.EventDispatcher] as well.

## Event handlers

Event handlers are attached to a compute step via its `add_event_handler`
method. Once attached, the handler receives every event the step emits.

The built-in [`ResultsHandler`][ropt.components.event_handlers.ResultsHandler],
[`HistoryHandler`][ropt.components.event_handlers.HistoryHandler], and
[`DataFrameHandler`][ropt.components.event_handlers.DataFrameHandler] are the same
objects you meet in [Running Optimizations](../running/running.md#built-in-handlers),
where they are described in full — there they are attached with
`optimize(handlers=...)`, here with `add_event_handler`, and they behave
identically. This section covers the underlying event model and the handlers
specific to workflows.

### Using handlers safely

An event handler is a stateful object that is **not safe for concurrent use**.
There are two ways to drive one, and they are mutually exclusive:

- **Attached directly to compute steps.** A handler may be attached to several
  compute steps, and a single instance can accumulate state across them — as
  long as those steps do not run it concurrently. Serial reuse is fine, even
  across different threads: each `handle_event` call must fully complete before
  the next begins. If two threads execute `handle_event` at the same time, a
  [`WorkflowError`][ropt.exceptions.WorkflowError] is raised.

- **Registered with an
  [`EventDispatcher`][ropt.components.event_handlers.EventDispatcher].** When work
  runs on several threads at once (for example, `ParallelEvaluator` with a
  multi-worker `ThreadingExecutor`), route events through a dispatcher. It
  receives events from any thread and delivers them to its handlers one at a
  time, so a single handler can safely aggregate results produced on many
  threads. See [Event Dispatcher](parallel.md#event-dispatcher) for the pattern.

A handler is owned by **either** one dispatcher **or** one-or-more compute
steps — never both — and may be registered with **at most one** dispatcher.
Mixing the two, or registering with a second dispatcher, raises a
[`WorkflowError`][ropt.exceptions.WorkflowError].

!!! note "A handler failure is fatal"

    An exception raised by a handler is a fatal error that stops the run. A
    directly-attached handler raises on the optimizer's own stack, so it
    propagates normally. A handler behind an
    [`EventDispatcher`][ropt.components.event_handlers.EventDispatcher] runs while
    the emitting run **waits** for the event to be handled, so its exception is
    re-raised on that run's own stack too — synchronously, including for the
    run's last event. Either way a handler bug surfaces as a single, clean
    exception — never a `BaseExceptionGroup`. See
    [Handler failures](parallel.md#handler-failures) for details.

!!! warning "Do not share a handler across parallel steps"

    Never attach the same handler instance to compute steps that may run at the
    same time on different threads; the moment a second thread executes it while
    the first is still inside `handle_event`, a
    [`WorkflowError`][ropt.exceptions.WorkflowError] is raised. Give
    each parallel step its own handler, or route events through an
    `EventDispatcher`.

    Serial reuse is allowed: the same handler may be reused across steps that
    run one after another, even on different threads, as long as their calls
    never overlap.

!!! note "Handlers are process-local"

    An event handler cannot be transferred to another process. Serializing one
    (for example when a task dispatched to a worker captures it) reconstructs it
    in the worker as an inert placeholder; `ropt` detects this and raises a
    [`TransferError`][ropt.exceptions.TransferError] before the task runs.
    Create handlers inside the worker and return their results
    as data. See
    [Nested workflows and process boundaries](parallel.md#nested-workflows-and-process-boundaries).

!!! note "Reading results is not thread-guarded"

    Handler state exposed through `handler[key]` is deliberately *not* bound to
    a thread, so results can be read after a run from any thread. Read a
    handler's stored values only **after its producer has finished**: after
    `step.run()` returns for a directly-attached handler, or after the
    [`EventDispatcher`][ropt.components.event_handlers.EventDispatcher] has been
    cancelled and its task group has exited for a handler registered with a
    dispatcher. Both are synchronization points that make the latest values
    visible.

    Reading a handler's state *while it is still processing events on another
    thread* returns a valid object, but possibly a stale one — do not rely on it
    for the latest result. For live progress during a parallel run, use a
    [`CallbackHandler`][ropt.components.event_handlers.CallbackHandler] (which is
    pushed each event) rather than polling another handler's state.

The result-collecting built-ins —
[`ResultsHandler`][ropt.components.event_handlers.ResultsHandler],
[`HistoryHandler`][ropt.components.event_handlers.HistoryHandler], and
[`DataFrameHandler`][ropt.components.event_handlers.DataFrameHandler] — are
described in full in [Running Optimizations](../running/running.md#built-in-handlers).
They expose their state through dictionary access (`handler[key]`);
`ResultsHandler` and `HistoryHandler` use the key `"results"`, while
`DataFrameHandler` uses the table name. At this level each also accepts a
`domain` argument (`"user"` — the default — or `"optimizer"`) that selects
whether results are transformed to the user domain before being stored; the
Simple API always uses the user domain.

Two more handlers exist only at this level, for wiring events:

| Handler                                                                    | Purpose                                                                                                          |
| -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| [`CallbackHandler`][ropt.components.event_handlers.CallbackHandler]        | Forward selected event types to a user callback.                                                                |
| [`EventForwardHandler`][ropt.components.event_handlers.EventForwardHandler]| Forward events to an [`EventDispatcher`][ropt.components.event_handlers.EventDispatcher] for lock-free dispatch. |

### CallbackHandler

[`CallbackHandler`][ropt.components.event_handlers.CallbackHandler] listens for
events and forwards them to a callback function. It is constructed with a set of
`event_types` to respond to and a single `callback`. When an event with a
matching type arrives, the callback is called with the
[`EnOptEvent`][ropt.events.EnOptEvent].

### EventForwardHandler

[`EventForwardHandler`][ropt.components.event_handlers.EventForwardHandler] is
attached to a compute step and forwards matching events to an
[`EventDispatcher`][ropt.components.event_handlers.EventDispatcher]. The dispatcher
dispatches them from the asyncio event loop's thread, so handlers registered on
the dispatcher require no locking.

See [Event Dispatcher](parallel.md#event-dispatcher) for the full pattern.

## Evaluators

A compute step never evaluates the model itself — it delegates to an
[`Evaluator`][ropt.components.evaluators.Evaluator] instance that you supply. The
available evaluators, and how to write the evaluation code they wrap, are covered
in [Writing Evaluation Callbacks](evaluation_callbacks.md). For parallel,
process-based, or HPC evaluation, see
[`ParallelEvaluator`][ropt.components.evaluators.ParallelEvaluator] in
[Parallel Evaluation](parallel.md).

## Where to next

- [Parallel Evaluation](parallel.md) — run evaluations off-process
  or on a cluster.
- [Building a Workflow](../tutorials/workflow.md) — step-by-step
  example building a workflow from scratch.
- Full example:
  [examples/advanced/workflow.py](https://github.com/TNO-ropt/ropt/blob/main/examples/advanced/workflow.py).
