# Parallel Evaluation

For non-trivial problems, function evaluations dominate runtime and are often
best run in parallel, either on a single machine or on a cluster. `ropt` uses
Python's `asyncio` framework to enable this.

This page assumes familiarity with [Optimization Workflows](workflows.md).

## Why asyncio?

A single compute step could in principle run its evaluations in parallel
without an event loop — for example by spawning threads directly. However,
the real power of an asynchronous approach emerges when **multiple compute
steps run concurrently**. With `asyncio`, several optimizations can share the
same pool of workers, the event loop dispatches evaluation tasks as they
arrive, and results flow back without blocking other work.

The [`ParallelEvaluator`][ropt.workflow.evaluators.ParallelEvaluator] is the
evaluator that bridges the synchronous compute-step `run()` call and the
asynchronous world. It submits individual evaluation tasks (one per row in the
variable batch) to an [`Executor`][ropt.workflow.executors.Executor] via an
`asyncio.Queue`. The executor picks tasks from the queue, runs them on its
workers, and places results into a results queue that the evaluator collects.

Because compute steps call `run()` synchronously, the step itself is typically
dispatched with `asyncio.to_thread` so the event loop remains free to service
the executor's workers and other concurrent steps.

## ParallelEvaluator

[`ParallelEvaluator`][ropt.workflow.evaluators.ParallelEvaluator] wraps a
per-realization function — the same kind of callable used by
[`FunctionEvaluator`][ropt.workflow.evaluators.FunctionEvaluator] — and submits
the rows of the evaluation batch as [`Task`][ropt.workflow.executors.Task]
objects to the executor's task queue. It then waits for results to arrive on a
results queue.

Constructor parameters:

| Parameter     | Description                                                          |
| ------------- | -------------------------------------------------------------------- |
| `function`    | Per-realization callable (same interface as `FunctionEvaluator`).    |
| `executor`    | The [`Executor`][ropt.workflow.executors.Executor] to dispatch tasks to. |
| `bundle_size` | Number of active evaluations to group into a single task (default: `1`). Use an integer `> 1` for a fixed maximum bundle size, or `0` to bundle all active evaluations of a batch into one task. |
| `queue_size`  | Maximum size of the results queue (0 = unlimited).                   |
| `get_name`    | Optional callable to generate a name for each task.                  |

By default each row of the variable batch is submitted as its own task. The
`bundle_size` parameter allows several active evaluations to be grouped into a
single task that the worker executes sequentially. This is useful when per-task
overhead (thread/process startup, HPC job submission) dominates the cost of an
individual evaluation, or when the total number of active evaluations in a batch
is much larger than the number of available workers.

The `get_name` callable, if provided, is called with the sequence of
[`EvaluationFunctionContext`][ropt.workflow.evaluators.EvaluationFunctionContext]
objects for every evaluation packed into the task (a single-element sequence
when `bundle_size=1`) and should return a single task name. When using the
`HPCExecutor`, names also serve as task identifiers and must be unique within a
batch. The returned name is stamped onto the `name` field of every
`EvaluationFunctionContext` in the task, so the user function can read
`context.name` to recover it.

If the executor is not running when `eval()` is called, the evaluator raises an
`Abort` exception.

## Executors

An [`Executor`][ropt.workflow.executors.Executor] manages an `asyncio.Queue` of
[`Task`][ropt.workflow.executors.Task] objects and dispatches them to a pool of
workers. All executors share the same lifecycle:

1. Create the executor instance.
2. Start it inside an `asyncio.TaskGroup` with `await executor.start(tg)`.
3. Use it (via `ParallelEvaluator` or `dispatch_tasks`).
4. Shut it down with `executor.cancel()`.

Three implementations are provided:

### ThreadingExecutor

[`ThreadingExecutor`][ropt.workflow.executors.ThreadingExecutor] dispatches
tasks to worker threads via `asyncio.to_thread`. Use this for I/O-bound
evaluations or when the evaluation function releases the GIL (e.g. calls into
C/Fortran).

| Parameter    | Description                                       |
| ------------ | ------------------------------------------------- |
| `workers`    | Number of concurrent worker threads (default: 1). |
| `queue_size` | Maximum task queue size (0 = unlimited).          |

### MultiprocessingExecutor

[`MultiprocessingExecutor`][ropt.workflow.executors.MultiprocessingExecutor]
uses a `ProcessPoolExecutor` with a `"spawn"` context. Use this for CPU-bound
evaluations where true parallelism is needed.

| Parameter             | Description                                                    |
| --------------------- | -------------------------------------------------------------- |
| `workers`             | Number of worker processes (default: 1).                       |
| `queue_size`          | Maximum task queue size (0 = unlimited).                       |
| `max_tasks_per_child` | Restart workers after this many tasks (default: `None` = never). Useful if evaluations leak memory, but adds significant overhead. |

#### Task serialization

Each task crosses a process boundary, so its function, arguments, and result
must be serialized. If [`cloudpickle`](https://github.com/cloudpipe/cloudpickle)
is installed (the `cloudpickle` extra), it is used for both directions: this
serializes lambdas, closures, and interactively-defined functions (such as those
written in a notebook cell) by value, so they can be used as task functions and
returned as results. Without `cloudpickle`, the executor falls back to the
standard `pickle` module, which requires task functions to be importable,
module-level objects; passing a lambda or closure then raises a `RuntimeError`
suggesting the `cloudpickle` extra.

#### The `__main__` guard

With the `"spawn"` start method, every worker process starts a fresh interpreter
that **re-imports the program's entry module** to rebuild its environment. If the
entry script creates or starts the executor at module top level, that re-import
runs the same code again in each worker, which tries to start yet more processes
before the interpreter has finished bootstrapping. Python aborts this, the
workers never start, and `MultiprocessingExecutor` raises a `RuntimeError` at
startup.

The fix is to keep the code that creates and runs the executor behind an
`if __name__ == "__main__":` guard (or inside a function called from there):

```python
# Wrong: created at module top level. Each worker re-imports this and fails.
asyncio.run(dispatch_tasks(functions, executor="multiprocessing"))
```

```python
# Right: the guarded block is skipped during the worker re-import.
if __name__ == "__main__":
    asyncio.run(dispatch_tasks(functions, executor="multiprocessing"))
```

This is the standard "safe importing of main module" contract of Python's
`multiprocessing`; it applies equally to
[`dispatch_tasks`](#dispatching-arbitrary-tasks) with
`executor="multiprocessing"`. Interactive sessions (Jupyter/IPython) and test
runners such as `pytest` are unaffected, because their entry module is
import-safe and is not re-executed on re-import.

A few less common issues cause the same startup error:

- **Re-import safety.** The worker re-runs the entry module's *top-level* code
  that sits outside the guard. Keep side-effecting statements — argument
  parsing, binding a socket/port, opening resources — inside functions or behind
  the guard, so re-importing the module in a worker is harmless.
- **Frozen applications.** When bundling with PyInstaller or cx_Freeze, call
  `multiprocessing.freeze_support()` as the first statement of the entry point;
  otherwise each worker re-launches the whole application.
- **Restricted environments.** An environment that cannot spawn processes — for
  example due to process, file-descriptor, or memory limits, or a container
  without shared-memory/semaphore support — will also fail this startup check.

### HPCExecutor

[`HPCExecutor`][ropt.workflow.executors.HPCExecutor] submits tasks as jobs to an
HPC scheduler (e.g. Slurm) via the `pysqa` library. Each task is serialized to
disk, submitted to the queue, polled for completion, and its result is
deserialized back. Requires `ropt[hpc]` to be installed.

The executor manages the full remote task lifecycle:

- Serializing the task (function and arguments) to a shared filesystem.
- Submitting the task as a job to the HPC queue.
- Polling the queue for the job's status.
- Retrieving results (or exceptions) once the job completes.

| Parameter     | Description                                                              |
| ------------- | ------------------------------------------------------------------------ |
| `workdir`     | Shared filesystem directory for temporary I/O files.                     |
| `workers`     | Maximum concurrent HPC jobs (default: 1).                                |
| `queue_size`  | Maximum task queue size (0 = unlimited).                                 |
| `interval`    | Polling interval in seconds (default: 1).                                |
| `queue_type`  | Queueing system type, e.g. `"slurm"` (default).                          |
| `template`    | Optional submission script template string.                              |
| `config_path` | Optional path to `pysqa` cluster configuration directory.                |
| `cluster`     | Optional cluster name (for multi-cluster installations).                 |
| `queue`       | Optional queue/partition name.                                           |
| `cores`       | CPUs per task (default: 1).                                              |

Configuration can be provided either via a `template` string or a `config_path`
directory containing `pysqa` configuration files. If neither is given, the
executor looks for a default configuration at:

```
<prefix>/share/ropt/pysqa/<queue_type>/
```

where `<prefix>` is the Python installation prefix (or system data prefix).
Find it with:

```python
from sysconfig import get_paths
print(get_paths()["data"])
```

This allows deployments to ship pre-configured cluster definitions by
installing them into `share/ropt/pysqa/` — no explicit `config_path`
argument is needed at runtime.

For multi-cluster `pysqa` configurations, the target cluster is resolved from
the `cluster` and `queue` arguments:

- If `cluster` is given, it is selected directly. When `queue` is also given,
  it must be available on that cluster.
- If only `queue` is given, the cluster that provides it is derived
  automatically. This requires exactly one cluster to provide the queue;
  otherwise (no match or multiple matches) an error is raised.

## Error handling

Executors and the [`ParallelEvaluator`][ropt.workflow.evaluators.ParallelEvaluator]
distinguish two classes of failure, and treat them very differently.

### Infrastructure failure (tolerated)

An *infrastructure* failure is one that is not caused by the evaluation function
itself: a worker process is killed (`BrokenProcessPool`), or an HPC job's output
file never appears or cannot be deserialized. These are delivered as an ordinary
result whose value is an [`ExecutorFailure`][ropt.exceptions.ExecutorFailure]
(via [`put_result`][ropt.workflow.executors.Task.put_result]). The evaluator
records the affected rows as failed realizations by writing `numpy.nan`. Such a
failure is *tolerated*: the optimization continues, and only aborts (with
`TOO_FEW_REALIZATIONS`) if too many realizations fail to satisfy the configured
minimum.

### User-code exception (aborts everything)

A *user-code* exception is one raised by the evaluation function itself — a bug
in the objective, a bad configuration, an unexpected input. This must not be
silently turned into a failed realization; it signals a genuine error the user
needs to see and fix. When the task function raises, the worker delivers the
exception on the results queue (via
[`put_error`][ropt.workflow.executors.Task.put_error], which also closes the
queue) **and** re-raises it into the executor's `asyncio.TaskGroup`. The two
channels play distinct roles:

- The queue item unblocks the owning
  [`ParallelEvaluator.eval`][ropt.workflow.evaluators.ParallelEvaluator.eval]
  call, which raises [`Abort`][ropt.exceptions.Abort] with
  `ExitCode.ABORT_FROM_ERROR`, chaining the original exception as the cause
  (`raise Abort(...) from exc`) so its message and traceback remain visible.
- The re-raise into the `TaskGroup` cancels sibling tasks. This is what makes
  "abort everything" work when an advanced user runs several compute steps
  concurrently in their own `asyncio.TaskGroup` sharing one executor: a genuine
  error in one objective propagates into that group, cancels the siblings, and
  surfaces with a traceback so they can fix and re-run.

For the [`HPCExecutor`][ropt.workflow.executors.HPCExecutor] the exception
crosses a process boundary. It is serialized with `cloudpickle`, which can
handle exception objects that the standard `pickle` module cannot, but does not
serialize tracebacks. The worker therefore attaches the formatted traceback as a
note (`exc.add_note(...)`) before serializing, so the originating traceback
travels with the exception. Exceptions that cannot be serialized at all are
wrapped in a `RuntimeError` carrying their `repr` and notes.

## Threads vs. processes: what crosses the boundary

The three executor types are not interchangeable: the choice does not only
affect performance, it determines what a dispatched compute step can still
*do*. One principle governs the difference.

- A **thread** shares memory with the process that started it. A step's control
  channels — the event handlers it invokes and the live asyncio loop, executors,
  and [`EventDispatcher`][ropt.workflow.event_handlers.EventDispatcher] it relies
  on — all keep working across threads within one process.
- A **process** — a
  [`MultiprocessingExecutor`][ropt.workflow.executors.MultiprocessingExecutor]
  worker or an [`HPCExecutor`][ropt.workflow.executors.HPCExecutor] job — shares
  none of that. It is **input/output only**: a task is serialized in and a
  result is serialized out, and nothing in between can reach back into the host
  process. This is deliberate, and it is enough for the common case — running an
  evaluation that produces a value the optimizer needs.

The rule that follows is: anything that must **communicate back** — emit events
to a dispatcher or *drive* a nested compute step — must stay **in the host
process**. A different *thread* is fine; a different *process* is not. Only
**self-contained, data-in / data-out** work belongs across a process boundary.

Two places where this matters in practice:

- **Nested optimization.** A step that runs an inner workflow must run
  in-process — sequentially or on a
  [`ThreadingExecutor`][ropt.workflow.executors.ThreadingExecutor] — while only
  the innermost leaf evaluations may go to a process or HPC worker. See
  [Nested workflows and process boundaries](#nested-workflows-and-process-boundaries).
- **Dispatching functions to workers.** A function sent to a process or HPC
  worker cannot use handlers or a dispatcher that live in the host process. If
  it runs an optimization there, that optimization must be self-contained and
  return its outcome as data. See
  [Dispatching arbitrary tasks](#dispatching-arbitrary-tasks).

`ropt` enforces the hard edge of this rule rather than leaving it to convention:
an [`OptimizationStep`][ropt.workflow.compute_steps.OptimizationStep] is bound to
its process and refuses to be transferred into a worker, as detailed under
[Nested workflows and process boundaries](#nested-workflows-and-process-boundaries).

## Dispatching arbitrary tasks

[`dispatch_tasks`][ropt.workflow.dispatch_tasks] is a utility function built on
top of the executor infrastructure. It runs an arbitrary collection of Python
callables in parallel — not necessarily as part of an optimization workflow. Use
it for one-off parallel work such as post-processing, ensemble replay, or any
batch computation that benefits from threading, multiprocessing, or HPC
submission.

It creates an executor internally based on the `executor` argument
(`"threading"`, `"multiprocessing"`, or `"hpc"`), submits all functions, and
returns the collected results.

`dispatch_tasks` is an `async` function — call it with `await` from an
asyncio context, or use `asyncio.run(dispatch_tasks(...))`:

```python
import asyncio
from ropt.workflow import dispatch_tasks

def task_a():
    return "result_a"

def task_b():
    return "result_b"

results = asyncio.run(dispatch_tasks([task_a, task_b], executor="threading"))
print(results)  # ["result_a", "result_b"]
```

The `functions` argument can be either a sequence of callables or a mapping
from name to callable. When a mapping is used, the keys serve as task names
(useful for identifying jobs on the HPC cluster).

| Parameter   | Description                                                         |
| ----------- | ------------------------------------------------------------------- |
| `functions` | Sequence or mapping of callables to execute.                        |
| `executor`  | Executor type: `"threading"`, `"multiprocessing"`, or `"hpc"`.      |
| `report`    | Optional callback invoked with each task result as it completes.    |
| `workers`   | Number of parallel workers (default: 4).                            |
| `workdir`   | Working directory for the HPC executor.                             |
| `cluster`   | Optional HPC cluster name.                                          |
| `queue`     | Optional HPC queue/partition name.                                  |
| `cores`     | CPUs per task for HPC (default: 1).                                 |

!!! note "Working directory"

    The dispatched functions cannot rely on the current directory being set
    consistently. Use absolute paths to read or write files. Setting the
    current directory in a `"threading"` executor affects all threads; in
    `"multiprocessing"` and `"hpc"` executors it can be changed safely per
    task.

!!! note "No event handling across process boundaries"

    Functions dispatched to a `"multiprocessing"` or `"hpc"` executor run in a
    separate process. If such a function runs a compute step, that step's event
    handlers stay in the worker process and cannot deliver events to a
    dispatcher or handler in the host process — return results as data instead.
    See [Event handling is a single-process mechanism](#event-dispatcher).

## Event dispatcher

When multiple compute steps run concurrently in worker threads, their event
handlers are called from multiple threads simultaneously. **Event handlers must
not be shared across concurrent compute steps**: doing so raises a `RuntimeError`.

[`EventDispatcher`][ropt.workflow.event_handlers.EventDispatcher] is the
required solution: it receives events on a queue and dispatches them to its own
handlers from the asyncio event loop's thread. Because all handler calls happen
on a single thread, handlers registered on the dispatcher are safe even when
events arrive from multiple concurrent steps.

This is especially useful when one set of handlers needs to aggregate results
from multiple concurrent compute steps.

`EventDispatcher` follows the same lifecycle as executors:

```python
async with asyncio.TaskGroup() as tg:
    executor = ThreadingExecutor(workers=4)
    await executor.start(tg)

    event_dispatcher = EventDispatcher()
    await event_dispatcher.start(tg)

    # Attach an EventForwardHandler to the compute step.
    step.add_event_handler(
        EventForwardHandler(
            event_dispatcher,
            event_types={EnOptEventType.FINISHED_EVALUATION},
        )
    )

    # Handlers registered on the dispatcher need no locking.
    result_handler = ResultsHandler()
    event_dispatcher.add_event_handler(result_handler)

    await asyncio.to_thread(step.run, variables=..., context=...)

    event_dispatcher.cancel()
    executor.cancel()
```

[`EventForwardHandler`][ropt.workflow.event_handlers.EventForwardHandler] is a
regular event handler that can be attached to a compute step. When invoked from
the worker thread it puts the event on the dispatcher's queue via a thread-safe
call. The dispatcher's processing loop then dispatches it to the registered
handlers.

!!! warning "Event handling is a single-process mechanism"

    An [`EventDispatcher`][ropt.workflow.event_handlers.EventDispatcher] and
    every [`EventHandler`][ropt.workflow.event_handlers.EventHandler] live in
    the process that created them. `EventForwardHandler` delivers events by
    calling the dispatcher's `put_event`, which schedules them onto its event
    loop with `call_soon_threadsafe` — a *thread*-safe call, not a
    *process*-safe one. A dispatcher reached from another process has no live
    loop, so a forwarded event cannot arrive. Rather than let it be silently
    dropped, forwarding an event or calling `put_event` from within a worker
    process raises a `RuntimeError`. Use
    [`is_worker_process`][ropt.workflow.executors.is_worker_process] — which
    returns `True` inside a
    [`MultiprocessingExecutor`][ropt.workflow.executors.MultiprocessingExecutor]
    or [`HPCExecutor`][ropt.workflow.executors.HPCExecutor] worker — to detect
    this context in your own code.

    Event handlers can therefore only observe events emitted **within their own
    process**. Any compute step executed out-of-process — for example a whole
    optimization sent to a
    [`MultiprocessingExecutor`][ropt.workflow.executors.MultiprocessingExecutor]
    or [`HPCExecutor`][ropt.workflow.executors.HPCExecutor], whether via
    [`dispatch_tasks`](#dispatching-arbitrary-tasks) or as the enclosing layer
    of a nested workflow — may attach handlers local to that worker process,
    but those handlers cannot deliver events to a dispatcher or handler in the
    host process. To collect information from out-of-process steps, return it as
    data (the task's return value, or result metadata) rather than through
    shared handlers.

    This is why process- and HPC-based parallelism belongs at the innermost
    (leaf) evaluations — which return data and emit no events — while any layer
    that drives event-producing compute steps must run in-process. See
    [Nested workflows and process boundaries](#nested-workflows-and-process-boundaries).

### Thread-based dispatch

By default, handlers registered with `EventDispatcher` are called directly in
the asyncio event loop's thread. This is efficient for handlers that only do
in-memory work, such as `ResultsHandler` or `HistoryHandler`.

If a handler performs blocking operations — writing results to a file, pushing
data to a database, sending over a network — pass `run_in_thread=True` when
registering it:

```python
event_dispatcher.add_event_handler(my_handler, run_in_thread=True)
```

`CallbackHandler` and `DataFrameHandler` (when a slow callback is set via
`set_callback`) are common cases where this is needed. When multiple handlers
with `run_in_thread=True` match the same event they are dispatched **in
parallel** via `asyncio.gather` — they do not block each other.

## Nested workflows and process boundaries

A *nested* workflow is a compute step whose evaluation function itself runs
another compute step — for example an outer optimizer whose objective is the
outcome of an inner optimization. As noted in [Why asyncio?](#why-asyncio),
several concurrent steps can share one asyncio event loop, and usually shared
[`Executor`][ropt.workflow.executors.Executor]s and an
[`EventDispatcher`][ropt.workflow.event_handlers.EventDispatcher] as well. All
of these live **in a single process**.

This is a consequence of the general rule that
[event handling is a single-process mechanism](#event-dispatcher): it places a
hard constraint on where each layer of a nested workflow may run:

!!! warning "The enclosing layer of a nested workflow must run in-process"

    The step that *runs* an inner workflow must execute in the same process as
    the shared event loop — dispatch it via a
    [`ThreadingExecutor`][ropt.workflow.executors.ThreadingExecutor], or run it
    synchronously. It cannot run inside a
    [`MultiprocessingExecutor`][ropt.workflow.executors.MultiprocessingExecutor]
    or [`HPCExecutor`][ropt.workflow.executors.HPCExecutor] worker, because a
    subprocess or HPC job has no access to the live loop, executors, or
    dispatcher. An inner
    [`ParallelEvaluator`][ropt.workflow.evaluators.ParallelEvaluator] running
    there would find `executor.loop is None` and raise `Abort`, and any events
    it emits would never reach the main-process dispatcher.

[`OptimizationStep`][ropt.workflow.compute_steps.OptimizationStep] enforces this
rule rather than leaving it to convention. **A step is bound to its process, not
to a thread.** The event handlers it invokes live in shared memory, so they keep
working across threads within one process but cannot cross a process boundary.
The invariant is therefore "a step lives in one process," *not* "a step must run
where it was created." Concretely:

- **Across threads (allowed).** A step may be created on one thread and run on
  another within the same process — for example created on the main thread and
  driven with `asyncio.to_thread` or a
  [`ThreadingExecutor`][ropt.workflow.executors.ThreadingExecutor] while a
  main-thread [`EventDispatcher`][ropt.workflow.event_handlers.EventDispatcher]
  collects its events. Event handling keeps working because memory is shared.
- **Across processes (forbidden).** A step refuses to be *transferred* into a
  [`MultiprocessingExecutor`][ropt.workflow.executors.MultiprocessingExecutor]
  or [`HPCExecutor`][ropt.workflow.executors.HPCExecutor] worker: unpickling one
  there raises a `RuntimeError`. Create the step **inside** the worker instead —
  a self-contained optimization that returns its result as data. A step created
  there is unknown to the host process and needs no cross-process communication.
- **Concurrently (forbidden).** A single step must not run more than once at a
  time: calling
  [`run`][ropt.workflow.compute_steps.OptimizationStep.run] on a step that is
  already running — for example from two threads — raises a `RuntimeError`. Give
  each concurrent optimization its own step; serial reuse of one step is fine.

Process- and HPC-based parallelism therefore belongs at the **innermost (leaf)
evaluations**, where the actual model runs — not at a layer that itself drives a
nested workflow. The nested examples follow exactly this shape:

- [`examples/nested.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/nested.py)
  — outer and inner optimizations run sequentially in the main process via
  `FunctionEvaluator`.
- [`examples/nested_multiprocess.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/nested_multiprocess.py)
  — outer optimizations run on a `ThreadingExecutor` (in-process); only the
  inner leaf evaluations run on a `MultiprocessingExecutor`.
- [`examples/nested_hpc.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/nested_hpc.py)
  — same pattern, with the inner leaf evaluations submitted to the cluster via
  `HPCExecutor`.

## Where to next

- Wire a parallel evaluator into a workflow:
  [Optimization Workflows](workflows.md).
- Reference: [Executors](../reference/workflow_executors.md),
  [Evaluators](../reference/workflow_evaluators.md).
