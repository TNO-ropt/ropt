# Sessions

The module-level [`optimize`][ropt.optimize] and [`evaluate`][ropt.evaluate]
evaluate one point at a time and return when they have finished. To access the
concurrency features of `ropt`, open a [`session`][ropt.session] and use its
methods to run optimizations that evaluate functions in parallel:

```python
from ropt import session

with session() as s:
    pool = s.thread_pool(workers=4)
    result = pool.optimize(config, x0, objective)
```

`pool.optimize` takes the same arguments as the module-level `optimize` and
returns the same result, but evaluates multiple functions in parallel.

## Why use a session

The module-level functions are enough for a single optimization that evaluates
one point at a time. A session adds the following:

- **Parallel evaluation.** Pools are built by a session; see [Pools](#pools).
- **Concurrent work.** [`Session.run`][ropt.Session.run] runs optimizations, or
  functions of your own, on threads at the same time, and returns their
  results. See [Running Multiple Optimizations](many_runs.md).
- **One abort for all work.** [`Session.abort`][ropt.Session.abort], called from
  another thread, stops everything the session runs.
- **Failure handling.** When one part fails, the session aborts the other parts
  and raises the exception in your script.
- **Cleanup.** When the `with` block ends, the session stops its work and
  releases its workers.

Concurrency alone does not require a session: module-level `optimize` calls
made from threads of your own run at the same time. They are then independent:
an abort or a failure in one does not reach the others.

### Session methods and module-level functions

A session also has [`optimize`][ropt.Session.optimize],
[`evaluate`][ropt.Session.evaluate] and
[`evaluate_batch`][ropt.Session.evaluate_batch]. Like the module-level
functions, they evaluate one point at a time on the calling thread. The
difference is that the optimization becomes part of the session:
`Session.abort` stops it, and a failure in it aborts the other work of the
session. A module-level `optimize` called from your script opens a session of
its own, so an abort of the enclosing session does not reach it.

Each call of `optimize`, `evaluate`, `evaluate_batch` or `run` on a session or
on one of its pools starts a **task**. Everything started from inside a task
belongs to the same session: evaluation functions, event handlers, report
callbacks, and functions given to `run`. This includes a module-level
`optimize` called from inside a task.

## Pools

A pool runs evaluations in parallel, on threads, in worker processes, or as
jobs on an HPC cluster. Pools are built by a session, with
[`thread_pool`][ropt.Session.thread_pool],
[`process_pool`][ropt.Session.process_pool] and
[`hpc_pool`][ropt.Session.hpc_pool], and the session releases their workers
when its block ends. Several pools can be open at once. See
[Evaluating in Parallel](parallel.md).

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
