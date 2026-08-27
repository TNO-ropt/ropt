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

The [`ParallelEvaluator`][ropt.components.evaluators.ParallelEvaluator] is the
evaluator that bridges the synchronous compute-step `run()` call and the
asynchronous world. It hands the rows of the variable batch to an
[`Executor`][ropt.components.executors.Executor] as a single
[`Submission`][ropt.components.executors.Submission]. The executor runs the
submission's work items on its workers and delivers each result back to the
submission, which the evaluator collects.

Because compute steps call `run()` synchronously, the step itself is typically
dispatched with `asyncio.to_thread` so the event loop remains free to service
the executor's workers and other concurrent steps.

## ParallelEvaluator

[`ParallelEvaluator`][ropt.components.evaluators.ParallelEvaluator] wraps a
per-realization function — the same kind of callable used by
[`FunctionEvaluator`][ropt.components.evaluators.FunctionEvaluator] — and submits
the rows of the evaluation batch as [`WorkItem`][ropt.components.executors.WorkItem]
objects in one [`Submission`][ropt.components.executors.Submission]. It then waits
for the submission's results.

Constructor parameters:

| Parameter     | Description                                                          |
| ------------- | -------------------------------------------------------------------- |
| `function`    | Per-realization callable (same interface as `FunctionEvaluator`).    |
| `executor`    | The [`Executor`][ropt.components.executors.Executor] to dispatch work to. |
| `bundle_size` | Number of active evaluations to group into a single work item (default: `1`). Use an integer `> 1` for a fixed maximum bundle size, or `0` to bundle all active evaluations of a batch into one work item. |
| `batch_id_callback` | Callable returning the next batch ID each time it is called (default: an internal [`BatchIdCounter`][ropt.components.evaluators.BatchIdCounter]). |

By default each row of the variable batch is submitted as its own task. The
`bundle_size` parameter allows several active evaluations to be grouped into a
single task that the worker executes sequentially. This is useful when per-task
overhead (thread/process startup, HPC job submission) dominates the cost of an
individual evaluation, or when the total number of active evaluations in a batch
is much larger than the number of available workers.

The work items it submits carry no name, so the `HPCExecutor` identifies each
with a generated UUID. Name them yourself by building the
[`WorkItem`][ropt.components.executors.WorkItem] objects and submitting them to
the executor directly.

If the executor is not running when `eval()` is called, the evaluator raises an
[`ExecutorStopped`][ropt.exceptions.ExecutorStopped] exception.

## Executors

An [`Executor`][ropt.components.executors.Executor] accepts
[`Submission`][ropt.components.executors.Submission] objects and dispatches their
[`WorkItem`][ropt.components.executors.WorkItem] objects to a pool of workers.
All executors share the same lifecycle:

1. Create the executor instance.
2. Start it inside an `asyncio.TaskGroup` with `await executor.start(tg)`.
3. Use it (via `ParallelEvaluator`, or by submitting
   [`Submission`][ropt.components.executors.Submission] objects directly).
4. Shut it down with `executor.cancel()` — see
   [Stopping an executor](#stopping-an-executor) for what that does to work
   that is already running.

One rule covers every executor that leaves the process: **out-of-process work
needs importable, module-level callables.** A work item's function and arguments
are serialized, and the standard `pickle` module can only send what it can look
up by name — so a lambda, a closure, or a function defined in a notebook cell is
refused with an [`ExecutionError`][ropt.exceptions.ExecutionError]. Installing
`ropt[cloudpickle]` lifts that restriction, and lifts it for the process, local
and HPC executors alike. `ThreadExecutor` serializes nothing and is never
affected.

!!! note "Working directory"

    Work items cannot rely on the current directory being set consistently.
    Use absolute paths to read or write files. Setting the current directory
    in a `ThreadExecutor` affects all threads; in any of the executors that run
    work in a process of its own it can be changed safely per work item.

!!! note "No event handling across process boundaries"

    A work item sent to a `ProcessExecutor`, a `LocalJobExecutor` or an
    `HPCExecutor` runs in
    a separate process. If such a work item runs a compute step, that step's
    event handlers stay in the worker process and cannot deliver events to a
    dispatcher or handler in the host process — return results as data instead.
    See [Event handling is a single-process mechanism](#event-dispatcher).

Four implementations are provided:

### ThreadExecutor

[`ThreadExecutor`][ropt.components.executors.ThreadExecutor] owns a
`ThreadPoolExecutor` of its own and dispatches tasks to it with
`loop.run_in_executor`. Use this for I/O-bound evaluations or when the
evaluation function releases the GIL (e.g. calls into C/Fortran).

The private pool is the point: `asyncio.to_thread` would use the event loop's
*shared* default executor, the same one every other `to_thread` call in the
process draws from — including the one that dispatches each compute step's
`run()`. Filling it with evaluations would starve the steps waiting on them,
which is precisely the deadlock
[`run_concurrent`][ropt.components.concurrency.run_concurrent] exists to avoid.

Threads are not a lesser form of parallelism here. Python runs one thread's
*bytecode* at a time, but a thread that is **waiting** holds nothing: the GIL is
released around blocking calls, and extension code may release it too. So
several optimizations sharing a `ThreadExecutor` genuinely overlap whenever
their evaluations wait — on a subprocess, a file, a socket — or spend their time
inside a library that has let the GIL go.

| Parameter    | Description                                       |
| ------------ | ------------------------------------------------- |
| `workers`    | Number of concurrent worker threads (default: 1). |

### ProcessExecutor

[`ProcessExecutor`][ropt.components.executors.ProcessExecutor]
uses a `ProcessPoolExecutor` with a `"spawn"` context. Use this for CPU-bound
evaluations where true parallelism is needed.

It is a narrow tool, not a general execution mode. It earns its cost in exactly
two cases: **pure-Python computation**, which threads cannot speed up, and
**isolating process-global state**, where each evaluation needs its own copy of
something a library keeps in module scope. Everything else it does — copying
arguments and results, re-importing the entry module in every worker, losing all
contact with the host process — is a price paid, not a feature. An evaluation
that mostly runs an external program gets no benefit from it whatsoever, and is
better served by a thread pool or by
[`LocalJobExecutor`][ropt.components.executors.LocalJobExecutor].

| Parameter             | Description                                                    |
| --------------------- | -------------------------------------------------------------- |
| `workers`             | Number of worker processes (default: 1).                       |
| `max_tasks_per_child` | Restart workers after this many work items (default: `None` = never). Useful if evaluations leak memory, but adds significant overhead. |

#### Work item serialization

Each work item crosses a process boundary, so its function, arguments, and result
must be serialized. If [`cloudpickle`](https://github.com/cloudpipe/cloudpickle)
is installed (the `cloudpickle` extra), it is used for both directions: this
serializes lambdas, closures, and interactively-defined functions (such as those
written in a notebook cell) by value, so they can be used as task functions and
returned as results. Without `cloudpickle`, the executor falls back to the
standard `pickle` module, which requires the task function and its arguments to
be importable, module-level objects; passing a lambda or closure then raises an
[`ExecutionError`][ropt.exceptions.ExecutionError] suggesting the `cloudpickle` extra.

#### The `__main__` guard

With the `"spawn"` start method, every worker process starts a fresh interpreter
that **re-imports the program's entry module** to rebuild its environment. If the
entry script creates or starts the executor at module top level, that re-import
runs the same code again in each worker, which tries to start yet more processes
before the interpreter has finished bootstrapping. Python aborts this, the
workers never start, and `ProcessExecutor` raises an
[`ExecutionError`][ropt.exceptions.ExecutionError] at startup.

The fix is to keep the code that creates and runs the executor behind an
`if __name__ == "__main__":` guard (or inside a function called from there):

```python
async def main():
    executor = ProcessExecutor(workers=4)
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        ...  # submit work here
        executor.cancel()
```

```python
# Wrong: run at module top level. Each worker re-imports this and fails.
asyncio.run(main())
```

```python
# Right: the guarded block is skipped during the worker re-import.
if __name__ == "__main__":
    asyncio.run(main())
```

This is the standard "safe importing of main module" contract of Python's
`multiprocessing`. Interactive sessions (Jupyter/IPython) and test
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

### LocalJobExecutor

[`LocalJobExecutor`][ropt.components.executors.LocalJobExecutor] runs each work
item as a separate process on the machine `ropt` is running on. It needs no
extras and no configuration.

It shares all of its machinery with
[`HPCExecutor`][ropt.components.executors.HPCExecutor] — the same `.in`/`.out`
files, the same poll loop, the same failure reporting — and differs only in what
starts a job. Where the HPC executor hands a command to a scheduler, this one
starts it directly.

| Parameter  | Description                                                                 |
| ---------- | --------------------------------------------------------------------------- |
| `workdir`  | Directory for each work item's files. The default is a temporary directory that this executor creates and removes again, unless there is something left in it to read. |
| `workers`  | Maximum concurrent local jobs (default: 1).                                 |
| `interval` | Polling interval in seconds (default: 0.1).                                 |
| `retries`  | Extra polls to wait for a result (default: 0).                              |
| `cleanup`  | Whether to remove a work item's files once it settles (default: `True`).    |

The defaults differ from the HPC ones for reasons that follow from where the
jobs run. `interval` is small because a local process is finished the moment it
exits, so waiting is dead time rather than politeness towards a scheduler.
`retries` is `0` because a job writes and renames its result before exiting, so
there is no shared filesystem that might not have caught up yet.

**Stopping kills the job's whole process group.** Each job is started in a
session of its own, so `cancel()` reaches whatever the job started itself,
rather than leaving those orphaned. This needs process groups, so the executor
is **POSIX only** and refuses to be constructed elsewhere.

Waiting for a killed job to actually die happens on a thread of its own, never
on the event loop: a Ctrl-C that has to wait for cancellation to finish is the
thing this arrangement avoids. That same thread removes a working directory this
executor created, once the jobs writing to it are gone.

It removes it only when there is nothing left in it to read, though. A work item
that failed keeps its captured output, and `cleanup=False` keeps everything, so
in either case the directory is **kept** and its path logged at `WARNING` — a
temporary directory has a random name, and one that is kept without being named
is one nobody can find. A directory you passed yourself is never removed.

Each run gets a directory of its own: an executor restarted after keeping one
creates another, rather than writing into the files it just handed over. Ask
[`workdir`][ropt.components.executors.LocalJobExecutor.workdir] for the current
one.

### HPCExecutor

[`HPCExecutor`][ropt.components.executors.HPCExecutor] submits tasks as jobs to an
HPC scheduler (e.g. Slurm) via the `pysqa` library. Each task is serialized to
disk, submitted to the queue, polled for completion, and its result is
deserialized back. Requires `ropt[hpc]` to be installed. Add `ropt[cloudpickle]`
to send functions the standard `pickle` module cannot.

The executor manages the full remote task lifecycle:

- Serializing the task (function and arguments) to a shared filesystem.
- Submitting the task as a job to the HPC queue.
- Polling the queue for the job's status.
- Retrieving results (or exceptions) once the job completes.
- Cancelling any jobs that are still outstanding when the executor stops.

Stopping the executor with `cancel()` asks the scheduler to delete every job it
has submitted, so an interrupted optimization does not leave orphan jobs behind
consuming the cluster allocation. Cancellation is best effort: if the scheduler
cannot be reached the failure is logged and stopping continues.

| Parameter     | Description                                                              |
| ------------- | ------------------------------------------------------------------------ |
| `workdir`     | Shared-filesystem directory for each work item's serialized I/O files. Required, and must be an absolute path. |
| `workers`     | Maximum concurrent HPC jobs (default: 1).                                |
| `interval`    | Polling interval in seconds (default: 1).                                |
| `queue_type`  | Queueing system type, e.g. `"slurm"` (default).                          |
| `template`    | Optional submission script template string.                              |
| `config_path` | Optional path to `pysqa` cluster configuration directory.                |
| `cluster`     | Optional cluster name (for multi-cluster installations).                 |
| `queue`       | Optional queue/partition name.                                           |
| `cores`       | CPUs per work item (default: 1).                                         |
| `retries`     | Extra polls to wait for a result that is missing or unreadable (default: 30). |
| `query_retries` | Extra attempts to query the scheduler after one fails (default: 30). |
| `cleanup`     | Whether to remove a work item's files once it settles (default: `True`). A failed work item keeps its captured output. |

A finished job's result is not always readable at once: the job may have died
before writing it, or the file may be caught half-written. `retries` is how many
**further** polls to allow before the work item is failed with an
[`ExecutorFailure`][ropt.exceptions.ExecutorFailure], so the grace period is
`retries × interval` seconds — 30 seconds with the defaults. `retries=0` gives up
on the first failed read.

`query_retries` bounds a separate failure: the scheduler itself being
unreachable. Once querying the queue has failed `query_retries + 1` times **in a
row**, every outstanding work item is failed rather than waited on for ever; an
answer in between starts the count again, so an occasional bad moment costs
nothing. The two are separate numbers because they are separate problems — a
shared filesystem that lags has nothing to do with a scheduler that is down.

A job that died before writing a result leaves its only trace in the `.txt` file
holding its captured output. That file is therefore **kept** when a work item
fails, even under `cleanup=True`, and its last lines are appended to the
`ExecutorFailure` message. This is what makes an error in the job itself — a
missing module, an unreadable input file, a scheduler rejection — visible at all,
since nothing else about it reaches the host process.

The job command runs `ropt.components.executors` as a module with
`sys.executable`, the interpreter that submitted it, rather than whatever
`python` the job's `PATH` happens to resolve to. Submitting from a virtual
environment therefore works without that environment being activated on the
compute node, provided the interpreter's path is valid there.

The `workdir` holds each work item's serialized `.in`/`.out` files (written at
absolute paths) and its captured stdout. It is also passed to `pysqa` as the
job's `working_directory`, which the standard scheduler templates turn into a
`chdir` directive (e.g. `#SBATCH --chdir=...`) — but whether a job actually runs
there depends on the submission template, so do not rely on it. Because work item
filenames derive from work item names, the executor **refuses to overwrite**
pre-existing files; give each concurrently-running executor its own
`workdir`.

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

## Stopping an executor

`executor.cancel()` stops the executor and returns immediately; it never waits
for work that is already running. Callers see the same thing whichever executor
they used — a submission still in progress ends with
[`ExecutorStopped`][ropt.exceptions.ExecutorStopped]. What differs is what keeps
running afterwards, because what *can* be done to running work differs:

| Executor | Work already running | Work not yet started |
| --- | --- | --- |
| `ThreadExecutor` | **runs to completion** | dropped |
| `ProcessExecutor` | the worker processes are **killed** | dropped |
| `LocalJobExecutor` | each job's **process group is killed** | dropped |
| `HPCExecutor` | the jobs are **deleted from the queue** | dropped |

**Threads run to completion because a thread cannot be cancelled.** Python
offers no way to interrupt one from outside, so an evaluation on a
`ThreadExecutor` decides for itself when it stops. `cancel()` returns at once,
but the program cannot leave until those evaluations return — the pool joins its
threads at interpreter shutdown. Rather than let that look like a hang, the
executor logs a `WARNING` naming how many are still running. An evaluation that
may run long and has to be interruptible belongs on one of the other three.

**The kill is a `SIGTERM`, and the guarantee is partial.** In each of the other
three cases the target is *asked* to end and is not waited for, so a process
that blocks or ignores `SIGTERM`, or that sits in an uninterruptible system
call, outlives the request. Stopping is a strong best effort — enough that an
interrupted program exits instead of waiting for the current batch — not a
promise that nothing of the run survives it.

!!! warning "A process pool orphans whatever an evaluation launched itself"

    [`ProcessExecutor`][ropt.components.executors.ProcessExecutor] terminates
    its own worker processes and nothing else. It installs no process groups, so
    a subprocess an evaluation started — a simulator, a solver, a shell
    pipeline — is never signalled: it keeps running, and is re-parented when the
    worker holding it dies. Nothing reports this, and the work simply continues
    after the program that asked for it has gone.

    [`LocalJobExecutor`][ropt.components.executors.LocalJobExecutor] is the
    backend that handles this, by giving each work item a session of its own and
    signalling the whole group. Use it when an evaluation launches external
    programs and stopping has to mean stopping.

### Ctrl-C

Ctrl-C raises `SIGINT`, which CPython turns into `KeyboardInterrupt` — but only
once the interrupted thread returns to the interpreter. A thread parked in
`Queue.get`, `Event.wait`, `Thread.join`, `Future.result` or `Lock.acquire` is
not there while it waits, and whether the signal breaks the wait depends on a
process-global flag, `SA_RESTART`.

CPython leaves that flag clear, so waits are interruptible. Some third-party
extension modules install their own `SIGINT` handler with the flag **set**, and
because it is process-global the effect is not confined to whoever set it: from
then on Ctrl-C appears to do nothing at all, for the entire program. The
symptom is identical on every backend, since the thread that fails to wake is
the one waiting for results rather than the one producing them.

`ropt` does not touch the flag. It is process-global state that belongs to the
program, and a library that quietly changes it decides for every other part of
that program as well — including the parts that wanted the handler they
installed. Nor would clearing it once be enough: any import that happens later
can set it again.

So this is left to you, and only if it happens to you. When it does,
[`restore_keyboard_interrupt`][ropt.utils.restore_keyboard_interrupt] clears the
flag, at the top of your script and after the imports:

```python
from ropt.utils import restore_keyboard_interrupt

restore_keyboard_interrupt()
```

See [Keyboard Interrupts](../utilities/keyboard_interrupt.md) for the whole
story.

### Platforms

[`LocalJobExecutor`][ropt.components.executors.LocalJobExecutor] needs process
groups and is **POSIX only**: constructing it anywhere else raises an
[`ExecutionError`][ropt.exceptions.ExecutionError] rather than quietly offering
a weaker kill.

The other three are not *known* to be broken on Windows, but they are not tested
there either. Two differences are certain: there are no process groups, so the
containment above does not exist, and there is no `SA_RESTART`, so the Ctrl-C
problem does not exist.

Free-threaded (no-GIL) builds of CPython are **untested and unsupported**.

## Error handling

Executors and the [`ParallelEvaluator`][ropt.components.evaluators.ParallelEvaluator]
distinguish two classes of failure, and treat them very differently.

### Infrastructure failure (tolerated)

An *infrastructure* failure is one that is not caused by the evaluation function
itself: a worker process is killed (`BrokenProcessPool`), or an HPC job's output
file never appears or cannot be deserialized. These are delivered as an ordinary
result whose value is an [`ExecutorFailure`][ropt.exceptions.ExecutorFailure]
(via [`deliver`][ropt.components.executors.Submission.deliver]). The evaluator
records the affected rows as failed realizations by writing `numpy.nan`. Such a
failure is *tolerated*: the optimization continues, and only aborts (with
`TOO_FEW_REALIZATIONS`) if too many realizations fail to satisfy the configured
minimum.

Only the `numpy.nan` survives that step — the `ExecutorFailure` and its message
do not reach the optimizer, so an aborted run reports `TOO_FEW_REALIZATIONS`
without saying why. The reason is logged instead, once per failed work item, at
`WARNING` from the `ropt.components.evaluators` logger; see
[Logging](../utilities/logging.md). Because `realization_min_success` defaults to
*all* realizations, a single failed work item is enough to end the run this way.

### User-code exception (raised)

A *user-code* exception is one raised by the evaluation function itself — a bug
in the objective, a bad configuration, an unexpected input. This must not be
silently turned into a failed realization; it signals a genuine error the user
needs to see and fix. When the work item's function raises, the worker ends the
submission with the exception (via
[`fail`][ropt.components.executors.Submission.fail]) and returns to serving
further work. It does **not** tear the executor down.

The owning
[`ParallelEvaluator.eval`][ropt.components.evaluators.ParallelEvaluator.eval]
call receives the exception and **re-raises the original unchanged**, aborting
the current evaluation. This is deliberately identical to the sequential
[`FunctionEvaluator`][ropt.components.evaluators.FunctionEvaluator]: whichever
evaluator is used, a bug in the objective surfaces as the same exception,
propagating out of `eval()` (and out of the compute step) on the calling thread.

Because the executor keeps running, its lifetime is owned by the **consumer's
scope**, not by the error:

- Left unhandled, the exception propagates out of the block that owns the
  executor (for example an `async with asyncio.TaskGroup()` that a compute step
  runs inside), whose unwinding cancels the workers and stops the executor —
  "abort everything".
- Caught before it reaches that block, the executor stays alive and can be
  reused for further evaluations. This is what lets several compute steps share
  one executor and lets a bug in one be isolated from the others.

Only a `BaseException` (for example a cancellation) still propagates out of the
worker directly, tearing the executor down — that is the intended teardown
signal and is left untouched.

For the [`HPCExecutor`][ropt.components.executors.HPCExecutor] the exception
crosses a process boundary. It is serialized with `cloudpickle` when that is
installed and with the standard `pickle` module otherwise; neither serializes
tracebacks. The job therefore attaches the formatted traceback as a note
(`exc.add_note(...)`) before serializing, so the originating traceback travels
with the exception. Exceptions that cannot be serialized at all are wrapped in a
`RuntimeError` carrying their `repr` and notes.

## Threads vs. processes: what crosses the boundary

The executor types are not interchangeable: the choice does not only
affect performance, it determines what a dispatched compute step can still
*do*. One principle governs the difference.

- A **thread** shares memory with the process that started it. A step's control
  channels — the event handlers it invokes and the live asyncio loop, executors,
  and [`EventDispatcher`][ropt.components.event_handlers.EventDispatcher] it relies
  on — all keep working across threads within one process.
- A **process** — a
  [`ProcessExecutor`][ropt.components.executors.ProcessExecutor]
  worker, a [`LocalJobExecutor`][ropt.components.executors.LocalJobExecutor] job
  or an [`HPCExecutor`][ropt.components.executors.HPCExecutor] job — shares
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
  [`ThreadExecutor`][ropt.components.executors.ThreadExecutor] — while only
  the innermost leaf evaluations may go to a process or HPC worker. See
  [Nested workflows and process boundaries](#nested-workflows-and-process-boundaries).
- **Dispatching functions to workers.** A function sent to a process or HPC
  worker cannot use handlers or a dispatcher that live in the host process. If
  it runs an optimization there, that optimization must be self-contained and
  return its outcome as data. See [Executors](#executors).

`ropt` enforces this at the process boundary. The workflow objects that hold
in-process state or a process-local communication channel — compute steps,
evaluators, event handlers, and the
[`EventDispatcher`][ropt.components.event_handlers.EventDispatcher] — may still be
serialized, but they do not survive the trip: on the worker side they are
reconstructed as inert placeholders that discard their original state. Before a
process or HPC worker runs a dispatched task, `ropt` checks whether such a
placeholder was transferred and, if so, raises a
[`TransferError`][ropt.exceptions.TransferError] instead of running the task
against a disconnected copy. See
[Nested workflows and process boundaries](#nested-workflows-and-process-boundaries)
for what belongs where.

## Event dispatcher

When multiple compute steps run concurrently in worker threads, their event
handlers are called from multiple threads simultaneously. **Event handlers must
not be shared across concurrent compute steps**: doing so raises a
[`WorkflowError`][ropt.exceptions.WorkflowError].

[`EventDispatcher`][ropt.components.event_handlers.EventDispatcher] is the
required solution: it receives events on a queue and dispatches them to its own
handlers from the asyncio event loop's thread. Because all handler calls happen
on a single thread, handlers registered on the dispatcher are safe even when
events arrive from multiple concurrent steps.

This is especially useful when one set of handlers needs to aggregate results
from multiple concurrent compute steps.

`EventDispatcher` follows the same lifecycle as executors:

```python
async with asyncio.TaskGroup() as tg:
    executor = ThreadExecutor(workers=4)
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

[`EventForwardHandler`][ropt.components.event_handlers.EventForwardHandler] is a
regular event handler that can be attached to a compute step. When the step
emits an event it submits the event to the dispatcher and **blocks on the
emitting run's own thread until every handler has processed it**, preserving the
order in which events are submitted. If a handler raises, the original exception
is re-raised there, on the run's stack (see [Handler failures](#handler-failures)).

!!! warning "Event handling is a single-process mechanism"

    An [`EventDispatcher`][ropt.components.event_handlers.EventDispatcher] and
    every [`EventHandler`][ropt.components.event_handlers.EventHandler] live in
    the process that created them. `EventForwardHandler` delivers events by
    calling the dispatcher's `dispatch_event`, which schedules them onto its
    event loop with `call_soon_threadsafe` — a *thread*-safe call, not a
    *process*-safe one. A dispatcher reached from another process has no live
    loop, so a forwarded event cannot arrive.

    Event handlers can therefore only observe events emitted **within their own
    process**. Any compute step executed out-of-process — for example a whole
    optimization sent to a
    [`ProcessExecutor`][ropt.components.executors.ProcessExecutor]
    or [`HPCExecutor`][ropt.components.executors.HPCExecutor], whether as a work
    item of its own or as the enclosing layer
    of a nested workflow — may attach handlers local to that worker process,
    but those handlers cannot deliver events to a dispatcher or handler in the
    host process. To collect information from out-of-process steps, return it as
    data (the task's return value, or result metadata) rather than through
    shared handlers.

    This is why process- and HPC-based parallelism belongs at the innermost
    (leaf) evaluations — which return data and emit no events — while any layer
    that drives event-producing compute steps must run in-process. See
    [Nested workflows and process boundaries](#nested-workflows-and-process-boundaries).

### Handler failures

Forwarding an event through an
[`EventForwardHandler`][ropt.components.event_handlers.EventForwardHandler] is
**synchronous**: the emitting run blocks until the dispatcher has run every
handler for that event. A handler failure is therefore delivered like an
evaluation error — on the emitting run's own call stack — mirroring the
executor's tolerated-vs-fatal split:

- An ordinary `Exception` from a handler is **re-raised on the emitting run's
  stack**, unwrapped — a single, clean exception that stops the run normally,
  exactly as if a directly-attached handler had raised inline. It is logged
  (with the handler and the event type) as it is caught. Because emission is
  synchronous, this covers the run's **last** event too; nothing is deferred or
  lost. Every handler for the event still runs before the error surfaces; if
  several fail, the first (in registration order) is raised.
- A `BaseException` (such as `CancelledError`) is **not** delivered this way: it
  remains the session teardown backstop and propagates, tearing the dispatcher
  task group down, as with the executor.

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

### Event throughput

A dispatcher processes its queue **one event at a time**: all handlers for an
event finish before the next event is taken. `run_in_thread=True` moves a
blocking handler off the event loop, but it does not overlap that handler with
the handlers of any *other* event — only with the threaded handlers of the same
event.

This serialization is deliberate.
[`EventHandler`][ropt.components.event_handlers.EventHandler] is not re-entrant,
and a handler shared by concurrently running optimizations — one accumulating
results across all of them, say — needs exactly this guarantee to stay
lock-free.

The price is that handler cost scales with the *total* number of events across
all runs sharing the dispatcher, and is paid on the critical path of every
event. Measured with eight concurrent runs emitting five events each, sharing
one dispatcher and one handler that blocks for 50 ms: **2.02 s elapsed**,
against 2.00 s for fully serial execution and 0.25 s if the handlers had run
fully concurrently — never more than one handler thread active at a time.

Runs share a dispatcher when they share a handler scope, which is the normal
case for [`optimize_many`][ropt.simple.optimize_many]: it reads the current
handler scope once on the calling thread and gives the same one to every job.
An N-way `optimize_many` therefore pays its handler cost serially. Keep shared
handlers cheap; if one must do heavy I/O, buffer in memory and flush once the
runs have finished.

## Two rules for using the low-level API

**Run a compute step in a worker thread.** `step.run()` is an ordinary
synchronous method that runs its evaluator and its handlers **on whatever
thread called it**. On the event loop thread it blocks the loop, and the
executors and dispatchers it is waiting for are tasks on that same loop:

```python
# Wrong: run() executes here, on the loop thread.
step.run(context=context, variables=x0)

# Right: the loop stays free to service the workers.
await asyncio.to_thread(step.run, context=context, variables=x0)
```

`ropt.simple` already does this for you; it applies when driving compute steps
yourself. A step that dispatches to an executor detects the mistake and raises
[`WorkflowError`][ropt.exceptions.WorkflowError] instead of hanging. A step that
only forwards events to a dispatcher on that same loop cannot detect it, so the
rule has to be followed rather than relied upon.

**Do not emit events from an event handler.** Events are processed one at a
time, so an event dispatched from inside a handler waits for the handler that
dispatched it. This is enforced, whether the handler runs on the event loop or
on the dispatcher's thread pool.

To feed one run's events into a shared dispatcher, register an
[`EventForwardHandler`][ropt.components.event_handlers.EventForwardHandler] on a
**compute step** rather than on a dispatcher.

## Nested workflows and process boundaries

A *nested* workflow is a compute step whose evaluation function itself runs
another compute step — for example an outer optimizer whose objective is the
outcome of an inner optimization. As noted in [Why asyncio?](#why-asyncio),
several concurrent steps can share one asyncio event loop, and usually shared
[`Executor`][ropt.components.executors.Executor]s and an
[`EventDispatcher`][ropt.components.event_handlers.EventDispatcher] as well. All
of these live **in a single process**.

This is a consequence of the general rule that
[event handling is a single-process mechanism](#event-dispatcher): it places a
hard constraint on where each layer of a nested workflow may run:

!!! warning "The enclosing layer of a nested workflow must run in-process"

    The step that *runs* an inner workflow must execute in the same process as
    the shared event loop — dispatch it via a
    [`ThreadExecutor`][ropt.components.executors.ThreadExecutor], or run it
    synchronously. It cannot run inside a
    [`ProcessExecutor`][ropt.components.executors.ProcessExecutor]
    or [`HPCExecutor`][ropt.components.executors.HPCExecutor] worker, because a
    subprocess or HPC job has no access to the live loop, executors, or
    dispatcher. An inner
    [`ParallelEvaluator`][ropt.components.evaluators.ParallelEvaluator] running
    there would find `executor.loop is None` and raise
    [`ExecutorStopped`][ropt.exceptions.ExecutorStopped], and any events
    it emits would never reach the main-process dispatcher.

[`OptimizationStep`][ropt.components.compute_steps.OptimizationStep] enforces this
rule: a step is **bound to its process, not to the thread that created it**. The
invariant is "a step lives in one process," not "a step runs where it was
created." Concretely:

- **Across threads (allowed).** A step may be created on one thread and run on
  another within the same process — for example created on the main thread and
  driven with `asyncio.to_thread` or a
  [`ThreadExecutor`][ropt.components.executors.ThreadExecutor] while a
  main-thread [`EventDispatcher`][ropt.components.event_handlers.EventDispatcher]
  collects its events. Event handling keeps working because memory is shared.
- **Across processes (forbidden).** A step must not be *transferred* into a
  [`ProcessExecutor`][ropt.components.executors.ProcessExecutor]
  or [`HPCExecutor`][ropt.components.executors.HPCExecutor] worker. Serializing one
  — for example when a dispatched task captures it — reconstructs it in the
  worker as an inert placeholder rather than a working copy whose events reach
  the host. `ropt` detects the transferred placeholder and raises a
  [`TransferError`][ropt.exceptions.TransferError] before running the task.
  Create the step **inside** the worker instead — a self-contained
  optimization that returns its result as data. A step created there is unknown
  to the host process and needs no cross-process communication.
- **Concurrently (forbidden).** A single step must not run more than once at a
  time: calling
  [`run`][ropt.components.compute_steps.OptimizationStep.run] on a step that is
  already running — for example from two threads — raises a
  [`WorkflowError`][ropt.exceptions.WorkflowError]. Give
  each concurrent optimization its own step; serial reuse of one step is fine.

Process- and HPC-based parallelism therefore belongs at the **innermost (leaf)
evaluations**, where the actual model runs — not at a layer that itself drives a
nested workflow. The nested examples follow exactly this shape:

- [`examples/advanced/nested.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/advanced/nested.py)
  — outer and inner optimizations run sequentially in the main process via
  `FunctionEvaluator`.
- [`examples/advanced/nested_multiprocess.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/advanced/nested_multiprocess.py)
  — outer optimizations run on a `ThreadExecutor` (in-process); only the
  inner leaf evaluations run on a `ProcessExecutor`.
- [`examples/advanced/nested_hpc.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/advanced/nested_hpc.py)
  — same pattern, with the inner leaf evaluations submitted to the cluster via
  `HPCExecutor`.

## Where to next

- Wire a parallel evaluator into a workflow:
  [Optimization Workflows](workflows.md).
- Reference: [Executors](../reference/components_executors.md),
  [Evaluators](../reference/components_evaluators.md).
