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

The [`AsyncEvaluator`][ropt.workflow.evaluators.AsyncEvaluator] is the
evaluator that bridges the synchronous compute-step `run()` call and the
asynchronous world. It submits individual evaluation tasks (one per row in
the variable batch) to a [`Server`][ropt.workflow.servers.Server] via an
`asyncio.Queue`. The server picks tasks from the queue, executes them using
its backend, and places results into a results queue that the evaluator
collects.

Because compute steps call `run()` synchronously, the step itself is
typically dispatched with `asyncio.to_thread` so the event loop remains free
to service the server's workers and other concurrent steps.

## AsyncEvaluator

[`AsyncEvaluator`][ropt.workflow.evaluators.AsyncEvaluator] wraps a
per-realization function — the same kind of callable used by
[`FunctionEvaluator`][ropt.workflow.evaluators.FunctionEvaluator] — and
submits each row of the evaluation batch as a separate
[`Task`][ropt.workflow.servers.Task] to the server's task queue. It then
waits for results to arrive on a results queue.

Constructor parameters:

| Parameter    | Description                                                          |
| ------------ | -------------------------------------------------------------------- |
| `function`   | Per-realization callable (same interface as `FunctionEvaluator`).    |
| `server`     | The [`Server`][ropt.workflow.servers.Server] to dispatch tasks to.   |
| `queue_size` | Maximum size of the results queue (0 = unlimited).                   |
| `get_name`   | Optional callable to generate a name for each task.                  |

The `get_name` callable, if provided, is called with an
[`EvaluatorFunctionContext`][ropt.workflow.evaluators.EvaluatorFunctionContext]
object and should return a string. When using the `HPCServer`, names also
serve as task identifiers and must be unique within a batch.

If the server is not running when `eval()` is called, the evaluator raises
an `Abort` exception.

## Servers

A [`Server`][ropt.workflow.servers.Server] manages an `asyncio.Queue` of
[`Task`][ropt.workflow.servers.Task] objects and dispatches them to a pool of
workers. All servers share the same lifecycle:

1. Create the server instance.
2. Start it inside an `asyncio.TaskGroup` with `await server.start(tg)`.
3. Use it (via `AsyncEvaluator` or `dispatch_tasks`).
4. Shut it down with `server.cancel()`.

Three implementations are provided:

### ThreadingServer

[`ThreadingServer`][ropt.workflow.servers.ThreadingServer] dispatches tasks to
worker threads via `asyncio.to_thread`. Use this for I/O-bound evaluations or
when the evaluation function releases the GIL (e.g. calls into C/Fortran).

| Parameter    | Description                                       |
| ------------ | ------------------------------------------------- |
| `workers`    | Number of concurrent worker threads (default: 1). |
| `queue_size` | Maximum task queue size (0 = unlimited).          |

### MultiprocessingServer

[`MultiprocessingServer`][ropt.workflow.servers.MultiprocessingServer] uses a
`ProcessPoolExecutor` with a `"spawn"` context. Use this for CPU-bound
evaluations where true parallelism is needed.

| Parameter             | Description                                                    |
| --------------------- | -------------------------------------------------------------- |
| `workers`             | Number of worker processes (default: 1).                       |
| `queue_size`          | Maximum task queue size (0 = unlimited).                       |
| `max_tasks_per_child` | Restart workers after this many tasks (default: `None` = never). Useful if evaluations leak memory, but adds significant overhead. |

### HPCServer

[`HPCServer`][ropt.workflow.servers.HPCServer] submits tasks as jobs to an HPC
scheduler (e.g. Slurm) via the `pysqa` library. Each task is serialized to
disk, submitted to the queue, polled for completion, and its result is
deserialized back. Requires `ropt[hpc]` to be installed.

The server manages the full remote task lifecycle:

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
| `queue_type`  | Queueing system type, e.g. `"slurm"` (default).                         |
| `template`    | Optional submission script template string.                              |
| `config_path` | Optional path to `pysqa` cluster configuration directory.                |
| `cluster`     | Optional cluster name (for multi-cluster installations).                 |
| `queue`       | Optional queue/partition name.                                           |
| `cores`       | CPUs per task (default: 1).                                              |

Configuration can be provided either via a `template` string or a
`config_path` directory containing `pysqa` configuration files. If neither
is given, the server looks for a default configuration at:

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

[`dispatch_tasks`][ropt.workflow.dispatch_tasks] is a utility function built
on top of the server infrastructure. It runs an arbitrary collection of
Python callables in parallel — not necessarily as part of an optimization
workflow. Use it for one-off parallel work such as post-processing, ensemble
replay, or any batch computation that benefits from threading,
multiprocessing, or HPC submission.

It creates a server internally based on the `server` argument
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

results = asyncio.run(dispatch_tasks([task_a, task_b], server="threading"))
print(results)  # ["result_a", "result_b"]
```

The `functions` argument can be either a sequence of callables or a mapping
from name to callable. When a mapping is used, the keys serve as task names
(useful for identifying jobs on the HPC cluster).

| Parameter   | Description                                                         |
| ----------- | ------------------------------------------------------------------- |
| `functions` | Sequence or mapping of callables to execute.                        |
| `server`    | Server type: `"threading"`, `"multiprocessing"`, or `"hpc"`.         |
| `report`    | Optional callback invoked with each task result as it completes.    |
| `workers`   | Number of parallel workers (default: 4).                            |
| `workdir`   | Working directory for the HPC server.                               |
| `cluster`   | Optional HPC cluster name.                                          |
| `queue`     | Optional HPC queue/partition name.                                  |
| `cores`     | CPUs per task for HPC (default: 1).                                 |

!!! note "Working directory"

    The dispatched functions cannot rely on the current directory being set
    consistently. Use absolute paths to read or write files. Setting the
    current directory in a `"threading"` server affects all threads; in
    `"multiprocessing"` and `"hpc"` servers it can be changed safely per
    task.

## Where to next

- Wire a parallel evaluator into a workflow:
  [Optimization Workflows](workflows.md).
- Reference: [Servers](../reference/workflow_servers.md),
  [Evaluators](../reference/workflow_evaluators.md).
