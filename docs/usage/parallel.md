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

## Event server

When multiple compute steps run concurrently in worker threads, their event
handlers are called from multiple threads simultaneously. **Event handlers must
not be shared across concurrent compute steps**: doing so raises a `RuntimeError`.

[`EventServer`][ropt.workflow.executors.EventServer] is the required solution:
it receives events on a queue and dispatches them to its own handlers from the
asyncio event loop's thread. Because all handler calls happen on a single
thread, handlers registered on the server are safe even when events arrive from
multiple concurrent steps.

This is especially useful when one set of handlers needs to aggregate results
from multiple concurrent compute steps.

`EventServer` follows the same lifecycle as evaluation executors:

```python
async with asyncio.TaskGroup() as tg:
    executor = ThreadingExecutor(workers=4)
    await executor.start(tg)

    event_server = EventServer()
    await event_server.start(tg)

    # Attach an EventForwardHandler to the compute step.
    step.add_event_handler(
        EventForwardHandler(
            event_server,
            event_types={EnOptEventType.FINISHED_EVALUATION},
        )
    )

    # Handlers registered on the server need no locking.
    result_handler = ResultsHandler()
    event_server.add_event_handler(result_handler)

    await asyncio.to_thread(step.run, variables=..., context=...)

    event_server.cancel()
    executor.cancel()
```

[`EventForwardHandler`][ropt.workflow.event_handlers.EventForwardHandler] is a
regular event handler that can be attached to a compute step. When invoked from
the worker thread it puts the event on the server's queue via a thread-safe
call. The server's processing loop then dispatches it to the registered
handlers.

### Thread-based dispatch

By default, handlers registered with `EventServer` are called directly in the
asyncio event loop's thread. This is efficient for handlers that only do
in-memory work, such as `ResultsHandler` or `HistoryHandler`.

If a handler performs blocking operations — writing results to a file, pushing
data to a database, sending over a network — pass `run_in_thread=True` when
registering it:

```python
event_server.add_event_handler(my_handler, run_in_thread=True)
```

`CallbackHandler` and `TableHandler` (when a slow callback is set via
`set_callback`) are common cases where this is needed. When multiple handlers
with `run_in_thread=True` match the same event they are dispatched **in
parallel** via `asyncio.gather` — they do not block each other.

## Where to next

- Wire a parallel evaluator into a workflow:
  [Optimization Workflows](workflows.md).
- Reference: [Executors](../reference/workflow_executors.md),
  [Evaluators](../reference/workflow_evaluators.md).
