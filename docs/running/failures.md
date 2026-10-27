# Failures and Aborts

When [`optimize`][ropt.optimize], [`evaluate`][ropt.evaluate],
[`evaluate_batch`][ropt.evaluate_batch] or [`run`][ropt.run] raises an exception,
you can catch it where you made the call. This page explains which other calls
are aborted and what happens to their running work.

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
        except MySimulationError:
            return float("nan")

    result = optimize(config, x0, objective, pool=s.thread_pool(workers=4))
```

`simulate` and `MySimulationError` stand for your own function and an exception
that it raises. When `simulate` raises `MySimulationError`, `run` raises it in
`objective`, which returns `NaN`. Nothing else is aborted: the optimization
counts that realization as failed, as it
does any `NaN`, and the other evaluations are not affected. How many failed
realizations a batch may contain is set by
[`realization_min_success`](../optimizer_setup/configuration_sections.md#realizations).

Without the `try`, `objective` does not catch the exception, and the
optimization fails. `optimize` aborts the calls started from its work, including the
`run` calls of the other evaluations and their simulations, and then raises
the `MySimulationError` in your script. If the exception is not caught inside
the `with` block, the session aborts the rest of its work.

## Where the exception is raised

`optimize`, `evaluate`, `evaluate_batch` and `run` raise the exception of a
failure in the code that called them. That code can catch the exception.
For example, `objective` catches the exception from `run` above. If it does
not catch it, the optimization fails in turn and raises the exception in
your script.

`run` raises the first exception raised by one of its functions, as soon as it
arrives, and the results of its other functions are lost; see
[Failure in one optimization](multiple_runs.md#failure-in-one-run) for a way to keep
them.

An optimization that stops with an exit code has not failed in this sense. One
that stops with `TOO_FEW_REALIZATIONS`, for instance, returns its result like any
other, and nothing is aborted.

## Which calls are aborted { #which-calls-are-aborted }

When a call fails, `ropt` aborts the calls started from its work. In the example
above, failure of the optimization aborts the `run` calls started by
`objective`. This applies at every depth, with or without a pool. When one
function given to `run` fails, its other functions and the calls they started
are aborted too.

Calls made from an evaluation, a callback or work given to `run` inherit this
relationship when they run in the same process. Calls made in process-pool
workers also receive the abort. HPC jobs are cancelled instead, as described
under [What happens to running work](#what-an-abort-does).

If your script catches the exception inside the session's `with` block, the
failure does not abort the session. Other independent calls keep running, and
your script can start more work using the same session. If the exception
escapes the block, the session closes and its pools cannot be reused.

### Failures and interrupts { #what-a-failure-aborts }

The failing call raises its original exception. Other calls aborted as a
consequence raise [`AbortedError`][ropt.exceptions.AbortedError] with
`exit_code` set to `ABORTED_ON_ERROR`. When an abort
propagates to a call started from the affected work, it keeps the same exit
code, even if that call uses another session's pool or no pool. Further calls
started from that work after the abort are aborted immediately.

An optimization or evaluation that cannot be built because its configuration is
invalid, or that is given arguments it rejects, such as a vector of the wrong
shape, raises the exception before it has started any work. The rejected call
itself therefore aborts nothing.

Trying to create a pool using a closed session, or passing one of its pools
to a call, raises [`WorkflowError`][ropt.exceptions.WorkflowError]. These
operations also abort nothing directly.

In both cases, if the exception is not caught inside an open session's `with`
block, that session ends and aborts its remaining work with `ABORTED`.

An exception raised by a call that is being aborted is still raised, but
aborts nothing further.

A `KeyboardInterrupt` or `SystemExit` is not a failure. When one passes through
a call, the calls started from its work, and for `run` its other functions, are aborted
with `ABORTED`, and the exception is raised. Other calls in the session keep
running if the exception is caught inside the session's `with` block.

If an exception raised inside the session's `with` block is not caught there,
whether it is a failure or a `KeyboardInterrupt`, the session ends and aborts
the calls that are still running with `ABORTED`.

### Aborting a session

When a session is aborted, so are the calls given its pools or made in its
`with` block, and the sessions opened there. Calling
[`Session.abort`](multiple_runs.md#stopping-from-outside) uses `USER_ABORT`;
leaving the session's `with` block uses `ABORTED`.
An aborted session stays aborted:
every call given one of its pools, or made in its `with` block, after the abort
is aborted at once. See
[Aborting from outside](multiple_runs.md#stopping-from-outside).

!!! note "Calls on threads you start are not aborted automatically"
    A call made from a thread that your own code starts does not inherit the
    abort of the call that started the thread, even if it starts inside an
    evaluation. Copy the context so that aborts propagate to its calls:

    ```python
    import contextvars
    import threading

    thread = threading.Thread(target=contextvars.copy_context().run, args=(work,))
    ```

    Calls on that thread can still be aborted through a session whose pool
    they use.

## What happens to running work { #what-an-abort-does }

An aborted call raises [`AbortedError`][ropt.exceptions.AbortedError], whose
`exit_code` is the exit code of the abort; see
[Exit Codes](../results/exit_codes.md). It returns no result and starts no
pending work. Whether it waits for work already running depends on the pool:

| Pool | Work already running when its call is aborted |
| --- | --- |
| none | runs to its end, and the call ends after it; `run` starts none of its remaining functions |
| `thread_pool` | runs to its end unless it [stops early](#stopping-early-when-aborted), but the call does not wait for it |
| `process_pool` | as with a thread pool, and its worker is killed if it is still running five seconds later |
| `hpc_pool` | its cluster job is cancelled; on the local queue, the job is terminated, together with every program it started, and killed five seconds later if still running |

Process workers receive the abort request, so calls made inside them
are aborted too. Calls inside an HPC job do not receive that request; instead,
the calling process requests cancellation of the job through the scheduler.

Without a pool, the optimization or evaluation runs the evaluation function
itself, so the evaluation in progress has to return first. Where the call does
not wait, an exception raised by work that is still running is logged as a
warning, unless it is an `AbortedError`.

`run`, an optimization and an evaluation also raise `AbortedError` when
their pool can no longer run the work. In an evaluation function, returning
`NaN` for that counts the realization as failed, so catch only the exceptions
that your own code raises, as the example above does.

## Stopping work cooperatively { #stopping-early-when-aborted }

[`aborting()`][ropt.aborting] lets running Python code detect an abort and
stop early. It works in code run by `ropt` **without a pool, on a thread pool,
and in process-pool workers**. It does not receive the calling process's
abort request inside an HPC job; those jobs are cancelled instead.

Check `aborting()` between steps of a long-running evaluation or a function
given to `run`. It returns `True` when the call running that code is being
aborted:

```python
from ropt import aborting

def work(steps):
    for step in steps:
        if aborting():
            return None
        step()
```

This check is especially useful on threads, which cannot be forcibly stopped.
In process-pool workers, the abort request becomes visible within a fraction
of a second, allowing the work to stop before its worker is killed.
Without a pool, the call must wait for the running function to return, so
checking `aborting()` can shorten that wait too.

An optimization, evaluation or `run` call started from work that is already
being aborted raises `AbortedError` immediately. Code that makes these calls
therefore needs no abort check of its own. An external program started with
[`run_program`][ropt.run_program] is terminated when the call that runs it is
aborted, and
`run_program` then raises `AbortedError`; see
[Running External Programs](programs.md).

In the `with` block of a session, outside any call, `aborting()` returns whether
the session is aborted. Outside a `ropt` call or session block it returns
`False`. This also applies inside an HPC job unless the job opens its own
session or starts its own `ropt` call: those can report their own aborts,
but not the abort of the call that submitted the job.
