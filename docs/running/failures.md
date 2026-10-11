# Failures and Aborts

`optimize`, `evaluate` and `evaluate_batch` each start one **task**, and
`offload`, on a [session][ropt.Session.offload] or a
[pool][ropt.WorkerPool.offload], starts one per function. Tasks can be under way
at the same time: side by side on one session, or started from inside one
another, as when an evaluation function calls `offload`. When a task raises an
exception, every task started from inside it is aborted, and so are the other
tasks started by the same `offload`. The exception is then raised in the code
that started the task. If the task was started from your script rather than from
inside another task, every other task on the session is aborted as well.

!!! note

    This page describes what a failure does to other tasks. What a single
    optimization raises or reports is described in
    [When something goes wrong](running.md#when-something-goes-wrong), and what
    happens when a worker breaks down in
    [When an evaluation fails](parallel.md#when-an-evaluation-fails).

Here an evaluation function offloads a simulation to a process pool:

```python
from functools import partial

from ropt import session

with session() as s:
    simulators = s.process_pool(workers=8)

    def objective(variables, context):
        try:
            return simulators.offload(partial(simulate, variables))
        except SimulationError:
            return float("nan")

    result = s.thread_pool(workers=4).optimize(config, x0, objective)
```

`simulate` and `SimulationError` stand for your own function and an exception
that it raises. `offload` is called from the evaluation function, so the task it
starts is [nested](#nested-tasks) in the optimization. When `simulate` raises
`SimulationError`, `offload` raises it in `objective`, which returns `NaN`. No
other task is aborted: the optimization counts that realization as failed, as it
does any `NaN`, and the other evaluations are not affected. How many failed
realizations a batch may contain is set by
[`realization_min_success`](../optimizer_setup/configuration_sections.md#realizations).

Without the `try`, the exception leaves `objective`, and the optimization fails
with it. The optimization was started from your script, so `optimize` aborts
every other task on the session, including the simulations that the other
evaluations had offloaded, and then raises the `SimulationError` in your script.

## Nested tasks { #nested-tasks }

A task started from inside another task is **nested** in it, and the outer task
is its **parent**. A task is started from inside another when `optimize`,
`evaluate`, `evaluate_batch` or `offload` is called, in the same process, from an
evaluation function, an event handler, a report callback, or a function that
`offload` runs. This holds for the module-level functions as well as for the
methods of a session or a pool.

An evaluation function runs in your process when its optimization or evaluation
has no pool or a thread pool, and so does a function offloaded to a thread pool
or with `Session.offload`.
On a process, local or HPC pool they run in another process, and the tasks they
start there are not nested. Neither is a task started from your script.

!!! note "Threads that your code starts"
    A task started from a thread that your own code starts is not nested, even
    when the thread is started from an evaluation function, so its failure aborts
    every other task on the session. Start the thread through
    `contextvars.copy_context().run` to make it nested:

    ```python
    import contextvars
    import threading

    thread = threading.Thread(target=contextvars.copy_context().run, args=(work,))
    ```

## Where the exception is raised

`optimize`, `evaluate`, `evaluate_batch` and `offload` raise the exception of a
failing task in the code that called them. For a nested task, that is the
evaluation function, event handler, report callback or offloaded function from
which it was started. That code can catch the exception. If it does not, the
task that the code belongs to fails in turn, and the same rules apply to that
task.

`offload` raises the first exception that arrives, as soon as it arrives, and the
results of its other functions are lost; see
[Failure in one optimization](many_runs.md#failure-in-one-run) for a way to keep
them.

An optimization that stops with an exit code has not failed in this sense. One
that stops with `TOO_FEW_REALIZATIONS`, for instance, returns its result like any
other, and no task is aborted.

## What a failure aborts

When a task raises an exception, these tasks are aborted:

- every task nested in it, at any depth;
- the other tasks started by the same `offload`;
- every other task on its session, unless the failing task is nested.

The same holds when an optimization or evaluation cannot be built because its
configuration is invalid, and when arguments are rejected before any task
starts, such as a vector of the wrong shape. A method called on a session that
has closed, or on one of its pools, raises
[`WorkflowError`][ropt.exceptions.WorkflowError] and aborts nothing. An
exception raised by a task that is being aborted is still raised, but aborts
nothing further.

A `KeyboardInterrupt` or `SystemExit` is not a failure. When one passes through
a task, the tasks nested in it and the other tasks of the same `offload` are
aborted with `ABORTED`, and the exception is raised. Other tasks on the session
are not aborted until the exception leaves the session's `with` block.

A failure does not abort the tasks of another session, unless they are nested in
one that it aborts. A module-level function such as [`optimize`][ropt.optimize]
opens a session of its own, which holds only the tasks started by that function.

## What an abort does

A nested task is aborted when its parent is, with the same exit code:
`USER_ABORT` after [`Session.abort`](running.md#stopping-from-outside), `ABORTED`
when the session's `with` block ends or a `KeyboardInterrupt` or `SystemExit`
passes through the parent, and `ABORTED_ON_ERROR` after a failure.
This also holds when the nested task belongs to another session, as one started
with the module-level `optimize` does. A nested task that starts after its
parent was aborted is aborted at once.

An aborted task raises [`AbortedError`][ropt.exceptions.AbortedError], whose
`exit_code` is the exit code of the abort; see
[Exit Codes](../results/exit_codes.md). It raises without waiting for the work it
has running, and returns no result. A function given to `offload` cannot be
stopped from outside. If it has not started, it is not run, and if it is
running, `offload` does not wait for it. What happens to work that is already
running depends on the pool:

| Pool | Work already running when its task is aborted |
| --- | --- |
| none | runs to its end, and the task ends after it |
| `thread_pool`, `process_pool`, or the threads of `Session.offload` | runs to its end, but the task does not wait for it |
| `local_pool` | is killed, together with everything it launched |
| `hpc_pool` | its cluster job is cancelled |

Without a pool, the evaluation function runs in the thread that started the
task, so the call in progress has to return first. Where the task does not wait,
an exception raised by work that is still running is logged as a warning, unless
it is an `AbortedError`.

`offload`, an optimization and an evaluation also raise `AbortedError` when
their pool can no longer run the work. In an evaluation function, returning
`NaN` for that counts the realization as failed, so catch only the exceptions
that your own code raises, as the example above does.

## Stopping early when aborted { #stopping-early-when-aborted }

Work that is left running after its task was aborted runs to its end, unless it
stops by itself. Call [`aborting`][ropt.aborting] in long-running code — an
evaluation function, an event handler, a report callback, or a function given to
`offload` — to find out whether the task that runs it is being aborted:

```python
from ropt import aborting

def work(steps):
    for step in steps:
        if aborting():
            return None
        step()
```

A task nested in an aborted task is aborted as well, so code that starts an
optimization, an evaluation or an `offload` needs no check of its own: the
nested task raises `AbortedError` at once. `aborting` returns `False` in code
that no task runs, which includes the worker processes of a process, local or
HPC pool.
