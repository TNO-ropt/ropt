# Sessions

Without a pool, [`optimize`][ropt.optimize] and [`evaluate`][ropt.evaluate]
evaluate one point at a time and return when they have finished. To access the
concurrency features of `ropt`, open a [`session`][ropt.session], build a pool
with it, and pass the pool to the functions that should evaluate in parallel:

```python
from ropt import optimize, session

with session() as s:
    pool = s.thread_pool(workers=4)
    result = optimize(config, x0, objective, pool=pool)
```

With a pool, `optimize` returns the same result, but evaluates multiple
functions in parallel on the pool's workers.

## Why use a session

The functions without a pool are enough for a single optimization that
evaluates one point at a time. A session adds the following:

- **Parallel evaluation.** Pools are built by a session; see [Pools](#pools).
- **Concurrent work.** [`run`][ropt.run], given a pool, runs optimizations, or
  functions of your own, on the pool's workers at the same time, and returns
  their results. See [Running Multiple Optimizations](many_runs.md).
- **One abort for all work.** [`Session.abort`][ropt.Session.abort], called from
  another thread, stops everything started with the session's pools or in its
  `with` block.
- **Cleanup.** When the `with` block ends, the session stops its work and
  releases its workers.

Concurrency alone does not require a session: `optimize` calls made from threads
of your own run at the same time. They are then independent: an abort or a
failure in one does not reach the others. This also holds for sessions opened in
different threads: they run independently.

Inside a session, start concurrent work with [`run`][ropt.run] and a pool rather
than with threads of your own. An exception raised in a thread of your own
stays in that thread: it does not leave the `with` block, so the session does
not abort its other work. A function called without a pool in such a thread is
not aborted with the session either. With `run`, the failure of one function
aborts the others, and the exception is raised in your script.

### Calls without a pool in a session

[`optimize`][ropt.optimize], [`evaluate`][ropt.evaluate],
[`evaluate_batch`][ropt.evaluate_batch] and [`run`][ropt.run] can be called
without a pool. They then work sequentially: the first three evaluate one point
at a time, and `run` calls its functions one after another.

Called in the session's `with` block, or from inside a call made there, these
calls are still aborted with the session: by
[`Session.abort`][ropt.Session.abort], or when an exception leaves the `with`
block. An aborted call raises [`AbortedError`][ropt.exceptions.AbortedError].

## Pools

A pool runs evaluations in parallel, on threads, in worker processes, or as
jobs on an HPC cluster. Pools are built by a session, with
[`thread_pool`][ropt.Session.thread_pool],
[`process_pool`][ropt.Session.process_pool] and
[`hpc_pool`][ropt.Session.hpc_pool], and the session releases their workers
when its block ends. Several pools can be open at once. See
[Evaluating in Parallel](parallel.md).

## Failures and aborts

When a call to `optimize`, `evaluate`, `evaluate_batch` or `run` raises an
exception, the calls made from inside it are aborted, and the exception is
raised in your script. If your script does not catch it, the exception leaves
the `with` block, and the session aborts everything that is still running.
[`Session.abort`][ropt.Session.abort], called from another thread, aborts every
call given one of its pools or made in its `with` block, and the session stays aborted.
Ctrl-C aborts them when the `KeyboardInterrupt` leaves the `with` block. An
aborted call raises
[`AbortedError`][ropt.exceptions.AbortedError]. See
[Failures and Aborts](failures.md) and
[Aborting from outside](many_runs.md#stopping-from-outside).

An abort stops the programs that the aborted calls started with
[`run_program`][ropt.run_program], and every program those programs started. See
[Running External Programs](programs.md).

## Closing the session

When the `with` block ends, the session aborts the calls that are still under
way and releases its pools. It then waits up to five seconds in total for
work running with its pools to end. Worker processes still busy after that are
killed, and threads are no longer waited for; see
[Stopping an optimization](parallel.md#stopping-a-run).

A session cannot be opened again once it has closed. Building a pool on a
closed session, or passing one of its pools to a call, raises
[`WorkflowError`][ropt.exceptions.WorkflowError]. A session may be opened
inside another, and the pools of different sessions are independent. A session
opened in the `with` block of another, or in a call made there, is aborted
with it.
