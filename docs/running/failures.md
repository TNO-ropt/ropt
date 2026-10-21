# Failures and Aborts

A call to [`optimize`][ropt.optimize], [`evaluate`][ropt.evaluate],
[`evaluate_batch`][ropt.evaluate_batch] or [`run`][ropt.run], with or without a
pool, can start further calls from inside its own work: from an evaluation function, an event handler, a
result callback or a function given to `run`. Those calls are **nested** in it.
When a call fails, every call nested in it is aborted, at any depth, and the
exception is raised in the code that made the call. For `run`, a failure of one
of its functions is a failure of the whole call, so its other functions are
aborted too. If no code catches the exception, it leaves the `with` block, and
the session aborts everything that is still running. Code that catches the
exception keeps the other calls running.

!!! note

    This page describes what a failure does to other calls. What a single
    optimization raises or reports is described in
    [When something goes wrong](running.md#when-something-goes-wrong), and what
    happens when a worker breaks down in
    [When an evaluation fails](parallel.md#when-an-evaluation-fails).

Here an evaluation function runs a simulation with a process pool:

```python
from functools import partial

from ropt import optimize, run, session

with session() as s:
    simulators = s.process_pool(workers=8)

    def objective(variables, context):
        try:
            return run(partial(simulate, variables), pool=simulators)
        except SimulationError:
            return float("nan")

    result = optimize(config, x0, objective, pool=s.thread_pool(workers=4))
```

`simulate` and `SimulationError` stand for your own function and an exception
that it raises. `run` is called from the evaluation function, so it is
[nested](#nested-calls) in the optimization. When `simulate` raises
`SimulationError`, `run` raises it in `objective`, which returns `NaN`. Nothing
else is aborted: the optimization counts that realization as failed, as it
does any `NaN`, and the other evaluations are not affected. How many failed
realizations a batch may contain is set by
[`realization_min_success`](../optimizer_setup/configuration_sections.md#realizations).

Without the `try`, the exception leaves `objective`, and the optimization fails
with it. `optimize` aborts the calls nested in it, including the `run` calls
of the other evaluations and their simulations, and then raises the `SimulationError`
in your script. The exception then leaves the `with` block, and the session
aborts the rest of its work.

## Nested calls { #nested-calls }

A call to `optimize`, `evaluate`, `evaluate_batch` or `run` is **nested** in
another such call when it is made, in the same process, from the work of that
other call: from its evaluation function, event handlers or result callback,
or from a function given to it if it is a `run` call. The outer call is its
**parent**. This holds with or without a pool.

An evaluation function runs in your process when its optimization or evaluation
has no pool or a thread pool, and so does a function given to `run` without a
pool or with a thread pool. With a process or HPC pool they run in another process.
A call they make in the worker of a process pool is aborted when the call that
runs them is; a call they make in an HPC job is not nested. Neither is a call
made from your script.

!!! note "Threads that your code starts"
    A call made from a thread that your own code starts is not nested, even
    when the thread is started from an evaluation function, so it is not aborted
    with the call that started the thread. Start the thread through
    `contextvars.copy_context().run` to make it nested:

    ```python
    import contextvars
    import threading

    thread = threading.Thread(target=contextvars.copy_context().run, args=(work,))
    ```

## Where the exception is raised

`optimize`, `evaluate`, `evaluate_batch` and `run` raise the exception of a
failure in the code that called them. For a nested call, that is the
evaluation function, event handler, result callback or function given to `run`
from which it was made. That code can catch the exception. If it does not, the
call that the code belongs to fails in turn, and the same rules apply to that
call.

`run` raises the first exception raised by one of its functions, as soon as it
arrives, and the results of its other functions are lost; see
[Failure in one optimization](many_runs.md#failure-in-one-run) for a way to keep
them.

An optimization that stops with an exit code has not failed in this sense. One
that stops with `TOO_FEW_REALIZATIONS`, for instance, returns its result like any
other, and nothing is aborted.

## What a failure aborts

When a call raises an exception, every call nested in it is aborted, at any
depth. When one function given to `run` raises, the `run` call fails, so its
other functions are aborted, together with the calls nested in them.

An optimization or evaluation that cannot be built because its configuration is
invalid, or that is given arguments it rejects, such as a vector of the wrong
shape, raises the exception before it has started anything to abort. Building a pool
with a session that has closed, or passing one of its pools to a call, raises
[`WorkflowError`][ropt.exceptions.WorkflowError] and aborts nothing. An
exception raised by a call that is being aborted is still raised, but aborts
nothing further.

A `KeyboardInterrupt` or `SystemExit` is not a failure. When one passes through
a call, the calls nested in it, and for `run` its other functions, are aborted
with `ABORTED`, and the exception is raised. Other calls in the session are not
aborted until the exception leaves the session's `with` block.

An exception that leaves the `with` block of a session, whether a failure or a
`KeyboardInterrupt`, ends the session, which aborts the calls that are still
running with `ABORTED`.

## What an abort does

A nested call is aborted when its parent is, with the same exit code:
`USER_ABORT` after [`Session.abort`](many_runs.md#stopping-from-outside), `ABORTED`
when the session's `with` block ends or a `KeyboardInterrupt` or `SystemExit`
passes through the parent, and `ABORTED_ON_ERROR` after a failure.
This also holds when the nested call is given a pool of another session, or no
pool at all. A nested call that is made after its parent was aborted is aborted
at once.

When a session is aborted, so are the calls given its pools or made in its
`with` block, and the sessions opened there. An aborted session stays aborted:
every call given one of its pools, or made in its `with` block, after the abort
is aborted at once. See
[Aborting from outside](many_runs.md#stopping-from-outside).

An aborted call raises [`AbortedError`][ropt.exceptions.AbortedError], whose
`exit_code` is the exit code of the abort; see
[Exit Codes](../results/exit_codes.md). It raises without waiting for the work it
has running, and returns no result. A function given to `run` that has not
started never starts, and if it is running with a pool, `run` does not wait for
it. What
happens to work that is already running depends on the pool:

| Pool | Work already running when its call is aborted |
| --- | --- |
| none | runs to its end, and the call ends after it; `run` starts none of its remaining functions |
| `thread_pool` | runs to its end unless it [stops early](#stopping-early-when-aborted), but the call does not wait for it |
| `process_pool` | as with a thread pool, and its worker is killed if it is still running five seconds later |
| `hpc_pool` | its cluster job is cancelled; on the local queue, the job is terminated, together with every program it started, and killed five seconds later if still running |

Without a pool, the optimization or evaluation runs the evaluation function
itself, so the evaluation in progress has to return first. Where the call does
not wait,
an exception raised by work that is still running is logged as a warning, unless
it is an `AbortedError`.

`run`, an optimization and an evaluation also raise `AbortedError` when
their pool can no longer run the work. In an evaluation function, returning
`NaN` for that counts the realization as failed, so catch only the exceptions
that your own code raises, as the example above does.

## Stopping early when aborted { #stopping-early-when-aborted }

Work that is left running after its call was aborted runs to its end, unless it
stops by itself or, with a process pool, its worker is killed five seconds later.
Call [`aborting`][ropt.aborting] in long-running code — an
evaluation function, an event handler, a result callback, or a function given to
`run` — to find out whether the call that runs it is being aborted:

```python
from ropt import aborting

def work(steps):
    for step in steps:
        if aborting():
            return None
        step()
```

A call nested in an aborted call is aborted as well, so code that starts an
optimization, an evaluation or a `run` call needs no check of its own: the
nested call raises `AbortedError` at once. An external program started with
[`run_program`][ropt.run_program] is terminated when the call that runs it is
aborted, and
`run_program` then raises `AbortedError`; see
[Running External Programs](programs.md).

In the `with` block of a session, outside any call, `aborting` returns whether
the session is aborted. It returns `False` outside any call and session block,
which includes the jobs of an HPC pool. In the worker processes of a process pool it turns `True` within a
fraction of a second of the abort.
