# Parallel Evaluation

For non-trivial problems, function evaluations dominate runtime and are commonly
run in parallel, either on a single machine or on a cluster. `ropt` does this
with **executors**: objects that run a batch of calls elsewhere and return the
results.

This page assumes familiarity with [Optimization Workflows](workflows.md).

## The blocking model

An [`Executor`][ropt.components.executors.Executor] has one method for work,
[`run`][ropt.components.executors.Executor.run]. It takes a sequence of
[`WorkItem`][ropt.components.executors.WorkItem] objects, blocks until every one
of them has an outcome, and returns the results in the order the items were
given. A compute step calls it on the thread it is already on.

Several compute steps can share one executor. Each blocks in its own `run()`
call on its own thread, and each gets back the results of its own batch. Running
several optimizations at once is therefore a matter of starting a thread per
optimization — [`run_concurrent`][ropt.components.concurrency.run_concurrent]
does that, and is what [`optimize_many`][ropt.simple.optimize_many] is built
on.

## ParallelEvaluator

[`ParallelEvaluator`][ropt.components.evaluators.ParallelEvaluator] wraps a
per-realization function — the same kind of callable used by
[`FunctionEvaluator`][ropt.components.evaluators.FunctionEvaluator] — and turns
each active row of the evaluation batch into its own
[`WorkItem`][ropt.components.executors.WorkItem]. It passes them all to
[`run`][ropt.components.executors.Executor.run] in one call and blocks until the
results come back.

Constructor parameters:

| Parameter     | Description                                                          |
| ------------- | -------------------------------------------------------------------- |
| `function`    | Per-realization callable (same interface as `FunctionEvaluator`).    |
| `executor`    | The [`Executor`][ropt.components.executors.Executor] to dispatch work to. |
| `batch_id_callback` | Callable returning the next batch ID each time it is called (default: the program-wide batch-ID counter). |
| `bundle_size` | Evaluations sent to a worker together, `0` for a whole batch, `None` for the executor's own default (default: `None`). |

How many of those work items travel to a worker together follows
`bundle_size`. See [Bundling](#bundling).

If the executor is closed when `eval()` is called, the evaluator raises a
[`WorkflowError`][ropt.exceptions.WorkflowError]; if it closes while the call is
waiting, an [`ExecutorStopped`][ropt.exceptions.ExecutorStopped] is raised
instead.

## Executors

An [`Executor`][ropt.components.executors.Executor] runs
[`WorkItem`][ropt.components.executors.WorkItem] objects on its workers. All
executors share the same lifecycle:

1. Create the executor instance.
2. Use it (via `ParallelEvaluator`, or by calling `run()` directly).
3. Release its workers with `close()`, or by using it as a context manager —
   see [Stopping an executor](#stopping-an-executor) for what that does to work
   that is already running.

An executor that is dropped without being closed still releases its workers when
it is collected: the thread and process pools shut down with the executor, and a
[`LocalJobExecutor`][ropt.components.executors.LocalJobExecutor] stops its
teardown thread and removes a temporary directory it created, if nothing is left
in it. The local case reports a `ResourceWarning`. Python ignores that category
by default; run with `-W default::ResourceWarning` or `-X dev` to see it.

One rule covers every executor that leaves the process: **out-of-process work
needs importable, module-level callables.** A work item's function and arguments
are serialized, and the standard `pickle` module can only send what it can look
up by name. A lambda or a closure cannot be looked up at all, so it is refused
before it is sent, with an
[`ExecutionError`][ropt.exceptions.ExecutionError]. Code defined in `__main__` —
a script you ran, a notebook cell, an interactive session — is different: it
*can* be looked up here, so it is sent by name, and the failure comes back from
the worker, which reports the name it could not find. Whether that name resolves
depends on the worker: `ProcessExecutor` re-imports `__main__`, so a script's
functions are found again, while the local and HPC executors run a fresh command
whose `__main__` is ropt's own, so they are not. Installing `ropt[cloudpickle]`
lifts the restriction for all of them. `ThreadExecutor` serializes nothing and
is never affected.

!!! note "Working directory"

    Work items cannot rely on the current directory being set consistently.
    Use absolute paths to read or write files. Setting the current directory
    in a `ThreadExecutor` affects all threads; in any of the executors that run
    work in a process of its own it can be changed safely per work item.

!!! note "No event handling across process boundaries"

    A work item sent to a `ProcessExecutor`, a `LocalJobExecutor` or an
    `HPCExecutor` runs in a separate process. If such a work item runs a compute
    step, that step's event handlers stay in the worker process and cannot
    deliver events to a handler in the host process — return results as data
    instead.
    See [Events are a single-process mechanism](workflows.md#events-are-a-single-process-mechanism).

### Bundling

A `bundle_size` says how many work items are sent to a worker together, to be
run one after another there. The default of `1` sends each item on its own,
spreading a batch as widely as the workers allow; a larger value amortizes the
cost of a transfer when the items are cheap relative to it; `0` sends a whole
batch at once. A bundle never spans batches, so `0` is bounded by one `run()`
call.

It can be set in two places. Every executor takes a `bundle_size` at
construction, used by any call that does not state one, and
[`run`][ropt.components.executors.Executor.run] takes one that overrides it for
that call. [`ParallelEvaluator`](#parallelevaluator) passes on the size it was
given, or `None` to leave the choice to the executor. Every executor honours it,
`ThreadExecutor` included: a bundle is one worker task there as well.

Four implementations are provided:

### ThreadExecutor

[`ThreadExecutor`][ropt.components.executors.ThreadExecutor] owns a private
`ThreadPoolExecutor` and runs each bundle as a task on it. Use this for I/O-bound
evaluations or when the evaluation function releases the GIL (e.g. calls into
C/Fortran).

The thread pool is private for a reason: a shared one would also hold the
threads that concurrent compute steps block on, and filling it with evaluations
would starve the steps waiting on them. That is the deadlock
[`run_concurrent`][ropt.components.concurrency.run_concurrent] exists to avoid.

Threads are not a lesser form of parallelism here. Python runs one thread's
*bytecode* at a time, but a thread that is **waiting** holds nothing: the GIL is
released around blocking calls, and extension code may release it too. So
several optimizations sharing a `ThreadExecutor` genuinely overlap whenever
their evaluations wait — on a subprocess, a file, a socket — or spend their time
inside a library that has let the GIL go.

| Parameter     | Description                                                          |
| ------------- | -------------------------------------------------------------------- |
| `workers`     | Number of concurrent worker threads (default: 1).                    |
| `bundle_size` | Default calls per worker task, `0` for a whole batch (default: 1).   |

### ProcessExecutor

[`ProcessExecutor`][ropt.components.executors.ProcessExecutor]
uses a `ProcessPoolExecutor` with a `"spawn"` context. Use this for CPU-bound
evaluations where true parallelism is needed.

Two cases call for it: **pure-Python computation**, which threads cannot speed
up, and **isolating process-global state**, where each evaluation needs its own
copy of something a library keeps in module scope. Its other effects — copying
arguments and results, re-importing the entry module in every worker, and losing
all contact with the host process — are costs of the process boundary. An
evaluation that mostly runs an external program gains nothing from it, since
waiting on that program already releases the GIL: a thread pool or
[`LocalJobExecutor`][ropt.components.executors.LocalJobExecutor] covers that
case.

| Parameter             | Description                                                    |
| --------------------- | -------------------------------------------------------------- |
| `workers`             | Number of worker processes (default: 1).                       |
| `max_tasks_per_child` | Restart workers after this many work items (default: `None` = never). Useful if evaluations leak memory, but adds significant overhead. |
| `bundle_size`         | Default calls per worker task, `0` for a whole batch (default: 1). |

#### Work item serialization

Each work item crosses a process boundary, so its function, arguments, and result
must be serialized. If [`cloudpickle`](https://github.com/cloudpipe/cloudpickle)
is installed (the `cloudpickle` extra), it is used to *write*: this serializes
lambdas, closures, and interactively-defined functions (such as those written in
a notebook cell) by value, so they can be used as task functions and returned as
results. Only writing has to choose. Reading is always the standard library's,
which is what `cloudpickle` itself uses.

Without `cloudpickle`, writing falls back to the standard `pickle` module, which
requires the task function and its arguments to be importable, module-level
objects. The two ways that can fail differ:

- A lambda or a closure cannot be written at all, so it is refused before it is
  sent, with an [`ExecutionError`][ropt.exceptions.ExecutionError] naming the
  extra.
- A notebook-defined function *can* be written, because it is stored by name and
  the name exists in your session. It fails in the worker, where that name does
  not resolve. The worker reports it as the error it is, with a note saying what
  could not be rebuilt.

!!! note "What a worker cannot report"

    The worker can only report a failure it survives to report. Anything that
    goes wrong before it reaches your work item — the worker failing to start,
    `ropt` failing to import, the entry module failing to re-import — still
    surfaces as a lost worker rather than a described error.

#### The `__main__` guard

With the `"spawn"` start method, every worker process starts a fresh interpreter
that **re-imports the program's entry module** to rebuild its environment. If the
entry script creates or starts the executor at module top level, that re-import
runs the same code again in each worker, which tries to start yet more processes
before the interpreter has finished bootstrapping. Python aborts this, the
workers never start, and `ProcessExecutor` raises an
[`ExecutionError`][ropt.exceptions.ExecutionError] at startup.

`ProcessExecutor` checks for the guard when it is constructed, so the failure
arrives at construction rather than at the first evaluation. The fix is to keep
the code that creates and uses the executor behind an
`if __name__ == "__main__":` guard (or inside a function called from there):

```python
def main():
    with ProcessExecutor(workers=4) as executor:
        ...  # run work here
```

```python
# Wrong: run at module top level. Each worker re-imports this and fails.
main()
```

```python
# Right: the guarded block is skipped during the worker re-import.
if __name__ == "__main__":
    main()
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
starts a job. Where the HPC executor submits a command to a scheduler, this one
starts it directly.

| Parameter  | Description                                                                 |
| ---------- | --------------------------------------------------------------------------- |
| `workdir`  | Directory for each work item's files. The default is a temporary directory that this executor creates and removes again, unless there is something left in it to read. |
| `workers`  | Maximum concurrent local jobs (default: 1).                                 |
| `interval` | Polling interval in seconds (default: 0.1).                                 |
| `retries`  | Extra polls to wait for a result (default: 0).                              |
| `cleanup`  | Whether to remove a work item's files once it settles (default: `True`).    |
| `bundle_size` | Default calls per job, `0` for a whole batch (default: 1).               |

The defaults differ from the HPC ones for reasons that follow from where the
jobs run. `interval` is small because a local process is finished the moment it
exits, so waiting is dead time rather than politeness towards a scheduler.
`retries` is `0` because a job writes and renames its result before exiting, so
there is no shared filesystem that might not have caught up yet.

**Stopping kills the job's whole process group.** Each job is started in a
session of its own, so `close()` reaches whatever the job started itself, rather
than leaving those orphaned. This needs process groups, so the executor is
**POSIX only** and refuses to be constructed elsewhere.

Waiting for a killed job to actually die happens on a thread of its own, so
`close()` returns without waiting it out. That same thread removes a working
directory this executor created, once the jobs writing to it are gone.

It removes it only when there is nothing left in it to read, though. A work item
that failed keeps its captured output, and `cleanup=False` keeps everything, so
in either case the directory is **kept** and its path logged at `WARNING` — a
temporary directory has a random name, and one that is kept without being named
is one nobody can find. A directory you passed yourself is never removed.

Each executor gets a directory of its own, rather than writing into the files of
a directory another one kept. Read
[`workdir`][ropt.components.executors.LocalJobExecutor.workdir] for it.

### HPCExecutor

[`HPCExecutor`][ropt.components.executors.HPCExecutor] submits tasks as jobs to an
HPC scheduler (e.g. Slurm) via the `pysqa` library. Each task is serialized to
disk, submitted to the queue, polled for completion, and its result is
deserialized back. Requires `ropt[hpc]` to be installed. Add `ropt[cloudpickle]`
to send functions the standard `pickle` module cannot.

Closing the executor asks the scheduler to delete every job it has submitted, so
an interrupted optimization does not leave orphan jobs behind consuming the
cluster allocation. Cancellation is best effort: if the scheduler cannot be
reached the failure is logged and closing continues.

| Parameter     | Description                                                              |
| ------------- | ------------------------------------------------------------------------ |
| `workdir`     | Shared-filesystem directory for each work item's serialized I/O files. Required, and must be an absolute path. |
| `workers`     | Maximum concurrent HPC jobs (default: 1).                                |
| `interval`    | Polling interval in seconds (default: 1).                                |
| `config_path` | The `pysqa` configuration directory. Defaults to the site-wide one installed alongside ropt. |
| `cluster`     | Optional cluster name (for multi-cluster installations). Defaults to the configured primary. |
| `queue`       | Optional name of a queue defined in the configuration. Defaults to the configured primary. |
| `template`    | A submission script template, submitted instead of any configuration.    |
| `scheduler`   | The queueing system a `template` is written for, e.g. `"slurm"` (default). Only meaningful with a `template`.|
| `cores`       | CPUs per work item (default: 1).                                         |
| `memory_max`  | Memory per work item. Rendered by the submission script, and clamped to the queue's limit when there is a configuration and the value is a number rather than a string. |
| `run_time_max` | Run time per work item, typically in seconds. Defaults to the queue's own limit. |
| `submit_options` | Extra variables for the submission script, for whatever it declares beyond the standard names. `None` entries are dropped. |
| `retries`     | Extra polls to wait for a result that is missing or unreadable (default: 30). |
| `query_retries` | Extra attempts to query the scheduler after one fails (default: 30). |
| `cleanup`     | Whether to remove a work item's files once it settles (default: `True`). A failed work item keeps its captured output. |

Jobs are described either by a `pysqa` configuration or by a `template`, and the
two are mutually exclusive; combining them raises a `ValueError` at
construction. Selecting a queue or a cluster, asking for resources and writing a
template are covered in
[Parallel Execution and Many Runs](../running/parallel.md#running-on-an-hpc-cluster).
What follows is the executor's own behaviour, and the layout of a configuration
directory.

A finished job's result is not always readable at once: the job may have died
before writing it, or the file may be caught half-written. `retries` is how many
**further** polls to allow before the work item is failed with an
[`ExecutorFailure`][ropt.components.executors.ExecutorFailure], so the grace
period is `retries × interval` seconds — 30 seconds with the defaults.
`retries=0` gives up on the first failed read.

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

The failure always names that file, even when its contents cannot be read. A
shared filesystem need not show them yet at the moment the work item is failed,
and a submission script that does not redirect the job's output — with
`#SBATCH --output={{output}}` or its equivalent — never writes them at all. The
message therefore points at the file whether or not it could quote from it.

The job command runs `ropt.components.executors` as a module with
`sys.executable`, the interpreter that submitted it, rather than whatever
`python` the job's `PATH` happens to resolve to. Submitting from a virtual
environment therefore works without that environment being activated on the
compute node, provided the interpreter's path is valid there.

The `workdir` holds each work item's serialized `.in`/`.out` files (written at
absolute paths) and its captured stdout. It is also passed to `pysqa` as the
job's `working_directory`, which the standard scheduler templates turn into a
`chdir` directive (e.g. `#SBATCH --chdir=...`) — but whether a job actually runs
there depends on the submission template, so do not rely on it. The files are
named after a generated id, so executors sharing a `workdir` do not collide on
them; a file already under that name is a leftover, and the executor
**refuses to overwrite** it.

#### Configuring the scheduler

`config_path` is the directory `pysqa` reads. Without it the executor uses the
site-wide configuration installed alongside ropt, at `<prefix>/share/ropt/pysqa/`,
where `<prefix>` is the Python installation prefix — deployments ship
pre-configured clusters by installing them there. Find it with:

```python
from sysconfig import get_paths
print(get_paths()["data"])
```

The directory holds a `queue.yaml` listing the queues, plus one submission
script per queue. A minimal Slurm configuration:

```yaml title="queue.yaml"
queue_type: SLURM
queue_primary: normal
queues:
  normal: {cores_max: 32, cores_min: 1, run_time_max: 3600, script: normal.sh}
  long:   {cores_max: 32, cores_min: 1, run_time_max: 86400, script: long.sh}
```

```jinja title="normal.sh"
#!/bin/bash
#SBATCH --partition=normal
#SBATCH --job-name={{job_name}}
#SBATCH --output={{output}}
#SBATCH --chdir={{working_directory}}
#SBATCH --ntasks={{cores}}
{%- if run_time_max %}
#SBATCH --time={{ [1, run_time_max // 60]|max }}
{%- endif %}
{%- if memory_max %}
#SBATCH --mem={{memory_max}}G
{%- endif %}

{{command}}
```

The scripts are
[Jinja](https://jinja.palletsprojects.com/en/stable/templates/) templates,
rendered by `pysqa` with `job_name`, `output`, `working_directory`, `cores`,
`memory_max`, `run_time_max` and `command`, plus whatever the caller passes in
`submit_options`. A variable the caller does not supply renders as empty, which
is why optional directives are wrapped in `{% if %}`. The partition is *not*
among them: it is written literally, which is why each queue normally needs its
own script.

Sites with more than one cluster use a `clusters.yaml` naming a `queue.yaml` per
cluster, each declaring its own `queue_type`; see the
[`pysqa` documentation](https://pysqa.readthedocs.io/en/latest/advanced.html#access-to-multiple-hpcs).

## Stopping an executor

`executor.close()` releases the executor's workers and returns immediately; it
never waits for work that is already running. A caller blocked in `run()` when
it happens is released with
[`ExecutorStopped`][ropt.exceptions.ExecutorStopped]; a `run()` started after it
is refused with a [`WorkflowError`][ropt.exceptions.WorkflowError]. What differs
per executor is what keeps running afterwards, because what *can* be done to
running work differs:

| Executor | Work already running | Work not yet started |
| --- | --- | --- |
| `ThreadExecutor` | **runs to completion** | dropped |
| `ProcessExecutor` | the worker processes are **killed** | dropped |
| `LocalJobExecutor` | each job's **process group is killed** | dropped |
| `HPCExecutor` | the jobs are **deleted from the queue** | dropped |

**Threads run to completion because a thread cannot be cancelled.** Python
offers no way to interrupt one from outside, so an evaluation on a
`ThreadExecutor` stops only when it returns. `close()` returns at once, but the
program cannot leave until those evaluations return — the thread pool joins its
threads at interpreter shutdown. Rather than let that look like a hang, the
executor logs a `WARNING` naming how many are still running. An
evaluation that may run long and has to be interruptible belongs on one of the
other three.

**The kill is a `SIGTERM`, and the guarantee is partial.** In each of the other
three cases the target is *asked* to end and is not waited for, so a process
that blocks or ignores `SIGTERM`, or that sits in an uninterruptible system
call, outlives the request. Stopping is a strong best effort — enough that an
interrupted program exits instead of waiting for the current batch — not a
promise that nothing of the run survives it.

!!! warning "A process executor orphans whatever an evaluation launched itself"

    [`ProcessExecutor`][ropt.components.executors.ProcessExecutor] terminates
    its own worker processes and nothing else. It installs no process groups, so
    a subprocess an evaluation started — a simulator, a solver, a shell
    pipeline — is never signalled: it keeps running, and is re-parented when the
    worker holding it dies. Nothing reports this, and the work continues
    after the program that asked for it has gone.

    [`LocalJobExecutor`][ropt.components.executors.LocalJobExecutor] is the
    backend that handles this, by giving each work item a session of its own and
    signalling the whole group. Use it when an evaluation launches external
    programs and stopping has to mean stopping.

### Ctrl-C

An imported extension module can set the process-global `SA_RESTART` flag, after
which Ctrl-C no longer breaks into a thread that is waiting. The thread waiting
for results is exactly the one that then fails to wake, so the symptom is
identical on every executor: Ctrl-C appears to do nothing at all. `ropt` does
not touch the flag, since it belongs to the program rather than to a library;
[`restore_keyboard_interrupt`][ropt.utils.restore_keyboard_interrupt] clears it
if you want it cleared. See
[Keyboard Interrupts](../troubleshooting/keyboard_interrupt.md) for the whole
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
distinguish two classes of failure. Both end the evaluation, but what the caller
receives, and what becomes of the executor, differ.

### Infrastructure failure (raised as `ExecutionError`)

An *infrastructure* failure is one that is not caused by the evaluation function
itself: a worker process is killed (`BrokenProcessPool`), or an HPC job's output
file never appears or cannot be deserialized. These are delivered as an ordinary
result whose value is an
[`ExecutorFailure`][ropt.components.executors.ExecutorFailure]
rather than raised, which leaves the executor usable rather than tearing it
down.

The evaluator turns that result into an
[`ExecutionError`][ropt.exceptions.ExecutionError], naming how many evaluations
were lost and why, which ends the run. It does **not** write `numpy.nan` for the
affected rows. A machine that broke is not a realization that failed to
converge: absorbing it would let the optimization continue on whichever workers
happened to survive, and produce a result that is indistinguishable from one
computed over the whole ensemble.

!!! note "A failed realization is still tolerated"

    `numpy.nan` returned *by the evaluation function* keeps its meaning — that
    realization could not produce a value — and
    [`realization_min_success`](../optimizer_setup/configuration_sections.md#realizations)
    still sets how many a batch may contain before the run ends with
    `TOO_FEW_REALIZATIONS`. Only a failure of the machinery is raised.

### User-code exception (re-raised unchanged)

A *user-code* exception is one raised by the evaluation function itself — a bug
in the objective, a bad configuration, an unexpected input. This must not be
silently turned into a failed realization; it signals a genuine error the user
needs to see and fix. When the work item's function raises, the worker abandons
the rest of that batch, hands the exception to the caller of `run()` and returns
to serving further work. It does **not** tear the executor down.

The owning
[`ParallelEvaluator.eval`][ropt.components.evaluators.Evaluator.eval]
call receives the exception and **re-raises the original unchanged**, aborting
the current evaluation. This is deliberately identical to the sequential
[`FunctionEvaluator`][ropt.components.evaluators.FunctionEvaluator]: whichever
evaluator is used, a bug in the objective surfaces as the same exception,
propagating out of `eval()` (and out of the compute step) on the calling thread.

Because the executor keeps running, its lifetime is owned by the **consumer's
scope**, not by the error:

- Left unhandled, the exception propagates out of the block that owns the
  executor — a `with` statement on it, for example — whose unwinding closes it.
- Caught before it reaches that block, the executor stays open and can be
  reused for further evaluations. This is what lets several compute steps share
  one executor and lets a bug in one be isolated from the others.

A `BaseException` — a `KeyboardInterrupt`, say — travels the same way and
reaches the caller unwrapped, rather than in a `BaseExceptionGroup`.

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
  channels — the event handlers it invokes and the executors it relies on — all
  keep working across threads within one process.
- A **process** — a
  [`ProcessExecutor`][ropt.components.executors.ProcessExecutor]
  worker, a [`LocalJobExecutor`][ropt.components.executors.LocalJobExecutor] job
  or an [`HPCExecutor`][ropt.components.executors.HPCExecutor] job — shares
  none of that. It is **input/output only**: a task is serialized in and a
  result is serialized out, and nothing in between can reach back into the host
  process. This is deliberate, and it is enough for the common case — running an
  evaluation that produces a value the optimizer needs.

The rule that follows is: anything that must **communicate back** — emit events
to a handler or *drive* a nested compute step — must stay **in the host
process**. A different *thread* is fine; a different *process* is not. Only
**self-contained, data-in / data-out** work belongs across a process boundary.

Two places where this matters in practice:

- **Nested optimization.** A step that runs an inner workflow must run
  in-process — sequentially or on a
  [`ThreadExecutor`][ropt.components.executors.ThreadExecutor] — while only
  the innermost leaf evaluations may go to a process or HPC worker. See
  [Nested workflows and process boundaries](#nested-workflows-and-process-boundaries).
- **Dispatching functions to workers.** A function sent to a process or HPC
  worker cannot use handlers that live in the host process. If
  it runs an optimization there, that optimization must be self-contained and
  return its outcome as data. See [Executors](#executors).

`ropt` enforces this at the process boundary. The workflow objects that hold
in-process state or a process-local communication channel — compute steps,
evaluators, and event handlers — each own a lock, so none of them can be
serialized at all. A task that captures one is
refused when it is submitted, with an
[`ExecutionError`][ropt.exceptions.ExecutionError] that names the object rather
than the lock it was caught on. The failure happens in the process that
submitted the work, before any worker sees it. See
[Nested workflows and process boundaries](#nested-workflows-and-process-boundaries)
for what belongs where.

## Two rules for using the low-level API

**Do not submit to an executor from one of its own workers.** A step running on
a [`ThreadExecutor`][ropt.components.executors.ThreadExecutor] worker occupies
that worker for as long as it waits, so work it submits to the same executor can
only start once it stops waiting — with one worker that is a deadlock, and with
several it is one whenever the waiting steps outnumber the free workers. `run()`
detects the caller and raises
[`WorkflowError`][ropt.exceptions.WorkflowError] instead of hanging. Give the
inner work an executor of its own; see
[Two executors, not one](../running/nested.md#two-executors-not-one).

**Do not run a compute step from inside a handler that the step can reach.** A
handler holds its own lock while `_handle_event` runs, so a step started there
that is attached to the same handler re-enters it. See
[Two hazards](workflows.md#two-hazards).

## Nested workflows and process boundaries

A *nested* workflow is a compute step whose evaluation function itself runs
another compute step — for example an outer optimizer whose objective is the
outcome of an inner optimization. Several concurrent steps usually share
[`Executor`][ropt.components.executors.Executor]s and event handlers, all of
which live **in a single process**.

This is a consequence of the general rule that
[events are a single-process mechanism](workflows.md#events-are-a-single-process-mechanism):
it places a hard constraint on where each layer of a nested workflow may run:

!!! warning "The enclosing layer of a nested workflow must run in-process"

    The step that *runs* an inner workflow must execute in the host process —
    dispatch it via a
    [`ThreadExecutor`][ropt.components.executors.ThreadExecutor], or run it
    synchronously. It cannot run inside a
    [`ProcessExecutor`][ropt.components.executors.ProcessExecutor]
    or [`HPCExecutor`][ropt.components.executors.HPCExecutor] worker, because a
    subprocess or HPC job has no access to the host process's executors, and any
    events it emits would never reach the main-process handlers.

[`OptimizationStep`][ropt.components.compute_steps.OptimizationStep] enforces this
rule: a step is **bound to its process, not to the thread that created it**. The
invariant is "a step lives in one process," not "a step runs where it was
created." Concretely:

- **Across threads (allowed).** A step may be created on one thread and run on
  another within the same process — for example created on the main thread and
  driven on a thread started by
  [`run_concurrent`][ropt.components.concurrency.run_concurrent] or on a
  [`ThreadExecutor`][ropt.components.executors.ThreadExecutor] worker, while
  handlers created on the main thread collect its events. Event handling keeps
  working because memory is shared.
- **Across processes (forbidden).** A step must not be *transferred* into a
  [`ProcessExecutor`][ropt.components.executors.ProcessExecutor]
  or [`HPCExecutor`][ropt.components.executors.HPCExecutor] worker. A step owns a
  lock, so serializing one — for example when a dispatched task captures it —
  fails, and the submission is refused with an
  [`ExecutionError`][ropt.exceptions.ExecutionError].
  Create the step **inside** the worker instead — a self-contained
  optimization that returns its result as data. A step created there is unknown
  to the host process and needs no cross-process communication.
- **Concurrently (forbidden).** A single step must not run more than once at a
  time: calling
  [`run`][ropt.components.compute_steps.ComputeStep.run] on a step that is
  already running — for example from two threads — raises a
  [`WorkflowError`][ropt.exceptions.WorkflowError]. Give
  each concurrent optimization its own step; serial reuse of one step is fine.

Process- and HPC-based parallelism therefore belongs at the **innermost (leaf)
evaluations**, where the actual model runs — not at a layer that itself drives a
nested workflow. The nested examples follow exactly this shape:

- [`examples/advanced/nested.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/advanced/nested.py)
  — outer and inner optimizations run sequentially in the main process via
  `FunctionEvaluator`.
- [`examples/advanced/nested_parallel.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/advanced/nested_parallel.py)
  — outer optimizations run on a `ThreadExecutor` (in-process); only the
  inner leaf evaluations run on a `ProcessExecutor`. Submitting those leaf
  evaluations to a cluster instead is the same shape, with `HPCExecutor` in
  place of `ProcessExecutor`.
