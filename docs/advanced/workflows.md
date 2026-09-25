# Optimization Workflows

!!! note

    This is one of two ways to **run** an optimization; the other is
    [Running Optimizations](../running/running.md). What the optimization does —
    its variables, objectives, constraints, and components — is set up in
    [Optimizer Setup](../optimizer_setup/key_concepts.md), the same whichever way you run
    it.

The workflow components are the layer beneath the
[simple API](../running/running.md) — the compute steps, event handlers,
evaluators and executors its convenience functions are assembled from, exposed
directly. Everything those functions do is available here, along with the cases
they cannot express; the cost is that you wire it together yourself.

These pages assume `asyncio` and threads. Parallel execution runs on an event
loop and event handlers may be invoked from several threads at once, so the
concurrency and process-boundary rules stated here are binding: breaking one
raises rather than misbehaving silently.

There are four core workflow components:

| Concept                                                                     | Role                                                                                            |
| --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| [`ComputeStep`][ropt.components.compute_steps.ComputeStep]                    | An executable unit of work (run an optimizer, run a single ensemble evaluation, etc.).          |
| [`EventHandler`][ropt.components.event_handlers.EventHandler]                 | A reactive object that observes events emitted by a compute step.                               |
| [`Evaluator`][ropt.components.evaluators.Evaluator]                           | The object a compute step uses to actually evaluate the model.                                  |
| [`Executor`][ropt.components.executors.Executor]                              | Dispatches evaluation tasks to threads, processes, or an HPC cluster.                           |

The first three are covered below. Executors are only relevant for asynchronous
and parallel execution and are discussed in [Parallel Evaluation](parallel.md).
Writing your own implementation of any of the four is covered in
[Implementing a Component](components.md).

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
print(f"Optimal variables: {result_handler['results'].variables}")
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
to call from a handler attached to several steps at once.

## Event handlers

Event handlers are attached to a compute step via its `add_event_handler`
method. Once attached, the handler receives every event the step emits.

The built-in [`ResultsHandler`][ropt.components.event_handlers.ResultsHandler],
[`HistoryHandler`][ropt.components.event_handlers.HistoryHandler], and
[`DataFrameHandler`][ropt.components.event_handlers.DataFrameHandler] are the same
objects you meet in [Result Handlers](../running/handlers.md#built-in-handlers),
where they are described in full — there they are attached with
`optimize(handlers=...)`, here with `add_event_handler`, and they behave
identically. This section covers the underlying event model and the handlers
specific to workflows.

### Using handlers safely

[`handle_event`][ropt.components.event_handlers.EventHandler.handle_event]
takes a lock around each call to the handler's `_handle_event`. A handler may
therefore be attached to any number of compute steps, including steps that run
at the same time on different threads: a second call waits for the first to
finish, and a single instance accumulates state across all of them. Handler
code sees one event at a time and needs no locking of its own.

The lock is not re-entrant. A call that reaches the same handler again while it
is still inside `_handle_event` raises a
[`WorkflowError`][ropt.exceptions.WorkflowError] if it is on the thread that
emitted the first event, and blocks on the lock if it is on another thread.
Both are described under [Two hazards](#two-hazards) below.

Handlers are called on the thread that emits the event, in the order they were
attached, and the emitting step waits until every handler for that event has
returned. A handler that blocks therefore holds up the run that emitted the
event, and any run waiting on the same handler's lock. The cost of a handler
scales with the number of events that reach it across every step it is attached
to. Keep a handler shared by concurrent runs cheap; if one must do heavy I/O,
buffer in memory and flush once the runs have finished.

!!! note "A handler failure is fatal"

    An exception raised by a handler is a fatal error that stops the run. It is
    raised on the optimizer's own stack, so it propagates normally, as a single
    exception — never a `BaseExceptionGroup`. The handlers attached after it do
    not see that event: `_emit_event` stops at the first failure.

### Two hazards

**Two handlers that touch the same external state.** Each handler is serialized
on its own, but not as a pair: a run can be inside one while another run is
inside the other. Take a [`threading.Lock`][threading.Lock] of your own inside
both `_handle_event` implementations. Each handler takes its own lock first and
the shared one second, so the acquisition order is the same for every caller.

Such a lock does not give the handlers an agreed order. If order matters, take
it from data in the event — `batch_id` and similar — rather than from the order
events arrive in, which depends on which run gets there first.

**A run started from inside a handler.** Nothing refuses this, and it is safe as
long as the run cannot reach a handler that is already running. If it does, the
handler is entered a second time while its lock is held. The outcome depends on
the run the handler starts, not on the call that is feeding the handler:

- A nested [`optimize`][ropt.simple.optimize], or a `step.run()` called from
  `_handle_event`, emits on the thread that called it — the one already inside
  the handler — so the re-entrancy check fires and a
  [`WorkflowError`][ropt.exceptions.WorkflowError] is raised.
- A nested [`optimize_many`][ropt.simple.optimize_many], or a nested
  [`run_concurrent`][ropt.components.concurrency.run_concurrent], emits on
  driver threads of its own, which wait for a lock the emitting thread holds
  until the nested run ends, and both stop.

Two handlers that each start a run reaching the other behave the same way:
raising when the cycle stays on one thread, blocking when it does not. Give a
nested run handlers of its own.

!!! note "Reading results is not thread-guarded"

    Handler state exposed through `handler[key]` is deliberately *not* bound to
    a thread, so results can be read after a run from any thread. Read a
    handler's stored values only **after its producer has finished**: after
    `step.run()` has returned for every step the handler is attached to. That
    return is the synchronization point that makes the latest values visible.

    Reading a handler's state *while it is still processing events on another
    thread* returns a valid object, but possibly a stale one — do not rely on it
    for the latest result. For live progress during a parallel run, use a
    [`CallbackHandler`][ropt.components.event_handlers.CallbackHandler] (which is
    pushed each event) rather than polling another handler's state.

The result-collecting built-ins —
[`ResultsHandler`][ropt.components.event_handlers.ResultsHandler],
[`HistoryHandler`][ropt.components.event_handlers.HistoryHandler], and
[`DataFrameHandler`][ropt.components.event_handlers.DataFrameHandler] — are
described in full in [Result Handlers](../running/handlers.md#built-in-handlers).
They expose their state through dictionary access (`handler[key]`);
`ResultsHandler` and `HistoryHandler` use the key `"results"`, while
`DataFrameHandler` uses the table name. Stored results carry both the configured
and the optimizer's domain; see
[Working with Results](../running/results.md#scaling-of-results).

One more handler exists only at this level, for wiring events:

| Handler                                                             | Purpose                                          |
| ------------------------------------------------------------------- | ------------------------------------------------ |
| [`CallbackHandler`][ropt.components.event_handlers.CallbackHandler] | Forward selected event types to a user callback. |

### CallbackHandler

[`CallbackHandler`][ropt.components.event_handlers.CallbackHandler] listens for
events and forwards them to a callback function. It is constructed with a set of
`event_types` to respond to and a single `callback`. When an event with a
matching type arrives, the callback is called with the
[`EnOptEvent`][ropt.events.EnOptEvent].

## Events are a single-process mechanism

Every [`EventHandler`][ropt.components.event_handlers.EventHandler] lives in the
process that created it, and is called on a thread of that process. Handlers
can therefore only observe events emitted **within their own process**.

A compute step executed out-of-process — for example a whole optimization sent
to a [`ProcessExecutor`][ropt.components.executors.ProcessExecutor] or
[`HPCExecutor`][ropt.components.executors.HPCExecutor], whether as a work item
of its own or as the enclosing layer of a nested workflow — may attach handlers
created inside that worker process, but those handlers cannot deliver anything
to a handler in the host process. To collect information from out-of-process
steps, return it as data: the task's return value, or result metadata.

This is why process- and HPC-based parallelism belongs at the innermost (leaf)
evaluations — which return data and emit no events — while any layer that
drives event-producing compute steps must run in-process. See
[Nested workflows and process boundaries](parallel.md#nested-workflows-and-process-boundaries).

A handler cannot be carried into a worker either. It holds a lock, so
serializing one — for example when a task dispatched to a worker captures it —
fails, and the submission is refused with an
[`ExecutionError`][ropt.exceptions.ExecutionError].

## Evaluators

A compute step never evaluates the model itself — it delegates to an
[`Evaluator`][ropt.components.evaluators.Evaluator] instance that you supply. The
available evaluators, and how to write the evaluation code they wrap, are covered
in [Writing Evaluation Callbacks](evaluation_callbacks.md). For parallel,
process-based, or HPC evaluation, see
[`ParallelEvaluator`][ropt.components.evaluators.ParallelEvaluator] in
[Parallel Evaluation](parallel.md).
