# Sessions

The module-level [`optimize`][ropt.optimize] and [`evaluate`][ropt.evaluate]
evaluate one point at a time and return when they have finished. For anything
more, open a **session** with [`session`][ropt.session] and start the work on
it. A session gives access to pools of workers, runs several optimizations at
once, aborts everything it runs when one part fails or when you ask it to, and
cleans up when its block ends:

```python
from ropt import session

with session() as s:
    pool = s.thread_pool(workers=4)
    result = pool.optimize(config, x0, objective)
```

`pool.optimize` takes the same arguments as the module-level `optimize` and
returns the same result. The session also has `optimize`, `evaluate` and
`evaluate_batch` methods of its own, which evaluate one point at a time, as
the module-level functions do.

## What belongs to a session

Each call of `optimize`, `evaluate`, `evaluate_batch` or `run` on a session or
on one of its pools starts a **task** that belongs to that session. What a
session does to its work, it does to these tasks, and to everything started
from inside them: an evaluation function, an event handler, a report callback,
or a function given to `run`.

Code that no task runs does not belong to the session, even inside its `with`
block. That includes the code of your script itself, and a module-level
function such as `optimize`, which opens a session of its own. An abort of the
outer session does not reach either.

## Pools

A pool runs evaluations in parallel, on threads, in worker processes, or as
jobs on an HPC cluster. Pools are built by a session, with
[`thread_pool`][ropt.Session.thread_pool],
[`process_pool`][ropt.Session.process_pool] and
[`hpc_pool`][ropt.Session.hpc_pool], and the session releases their workers
when its block ends. Several pools can be open at once. See
[Evaluating in Parallel](parallel.md).

## Several optimizations at once

[`Session.run`][ropt.Session.run] runs functions on threads of their own and
returns their results. Pass it several `optimize` calls and they run at the
same time. See [Several Optimizations at Once](many_runs.md).

## Failures and aborts

When a task started from your script raises an exception, the session aborts
its other tasks and raises the exception in your script.
[`Session.abort`][ropt.Session.abort], called from another thread, aborts every
task of the session, and Ctrl-C aborts them when the `KeyboardInterrupt` leaves
the `with` block. An aborted task raises
[`AbortedError`][ropt.exceptions.AbortedError]. See
[Failures and Aborts](failures.md) and
[Aborting from outside](many_runs.md#stopping-from-outside).

An abort stops a program that a task started with
[`run_program`][ropt.run_program], together with every program it started. See
[Running External Programs](programs.md).

## Closing the session

When the `with` block ends, the session aborts the tasks that are still under
way and releases its pools. It then waits up to five seconds in total for
running work to end. Worker processes still busy after that are killed, and
threads are no longer waited for; see
[Stopping an optimization](parallel.md#stopping-a-run).

A session cannot be opened again once it has closed. Calling a method of a
closed session or of one of its pools raises
[`WorkflowError`][ropt.exceptions.WorkflowError]. A session may be opened
inside another, and the pools of different sessions are independent.
