# Running Multiple Optimizations

Several optimizations can run in parallel on a thread, process or HPC pool.
Wrap each `optimize` call in a zero-argument callable and pass the callables to
[`run`][ropt.run] with the chosen pool. `run` starts each one on a worker of
that pool and returns the results in the order the callables were given.

The pool type determines what the optimizations can share:

- **Threads** run in the calling process and can share evaluation pools,
  handlers and other Python objects. Shared handlers handle one event at a
  time, but shared result callbacks and other mutable state need their own
  synchronization.
- **Processes and HPC jobs** run outside the calling process. The callables,
  their data and their return values must be serializable. Each callable must
  create any sessions, evaluation pools and handlers it needs in its worker;
  it cannot use those from the calling process. Return any collected results
  explicitly, since changes to objects in a worker do not update objects in
  the calling process.

The examples below use threads so that the optimizations can share an
evaluation pool and handlers. See
[Pools inside an evaluation](parallel.md#pools-inside-an-evaluation) for the
same restrictions when an evaluation starts an optimization.

!!! warning "Some backends need process isolation"
    When optimizations run on threads, their backends share one process. A
    backend plugin that uses process-wide state or fixed filenames may need
    [`external/...`](../optimizer_setup/optimizer.md#external-backend) to run
    safely alongside other optimizations. The built-in SciPy backend does not
    need this isolation. See
    [Isolating the optimizer](parallel.md#process-pool) for the distinction
    between isolating a backend and isolating its evaluations.

```python
from functools import partial

from ropt import optimize, run, session

with session() as s:
    pool = s.thread_pool(workers=4)
    runner = s.thread_pool(workers=len(start_points))
    optimizations = [
        partial(optimize, config, x0, objective, pool=pool)
        for x0 in start_points
    ]
    results = run(optimizations, pool=runner)
```

Each callable is an ordinary `optimize` call, so every optimization can have its
own configuration, initial values, objective, `metadata` and result callback.

The optimizations need a pool of their own, `runner` here, separate from the
pool they evaluate with. A call made from work running with a pool cannot use
that same pool: it raises [`WorkflowError`][ropt.exceptions.WorkflowError]. Without
a pool, `run` calls the functions one after another on the calling thread.

!!! tip "Give each optimization an ID"
    Pass `metadata` to tag each optimization with an identifier that is
    attached to its results (and shows up in a
    [`DataFrameHandler`](../results/handlers.md#dataframehandler)'s tables):

    ```python
    labels = ["low", "mid", "high"]
    results = run(
        [
            partial(
                optimize, config, x0, objective, pool=pool, metadata={"label": label}
            )
            for x0, label in zip(start_points, labels)
        ],
        pool=runner,
    )
    for result in results:
        print(result.results.metadata["label"])
    ```

    See [Attaching metadata](running.md#attaching-metadata) for details.

## Two levels of concurrency

There are two independent levels of concurrency here:

- **The optimizations** run concurrently, each on a worker of the pool given
  to `run`. The number of workers of that pool caps how many of them are under
  way at the same time; the others wait for a free worker.
- **The function evaluations** inside those optimizations use the pool
  passed to each `optimize` call, and the pool determines how they are
  parallelized. With `thread_pool(workers=1)` the optimizations still progress
  together, but their evaluations are executed one at a time. A larger pool —
  `thread_pool(workers=n)`, or a process or HPC pool — runs several
  evaluations at once.

Without an evaluation pool, each optimization calls the evaluation function
directly on the worker running that optimization. If `run` uses a thread pool,
several threads may therefore call the evaluation function at once. If it
cannot safely handle concurrent calls, give the optimizations a shared
evaluation pool with one worker, or a process or HPC pool that isolates the
evaluations.

One pool is one budget: `workers=10` means ten evaluations at a time across all
the optimizations that use the pool, not ten per optimization. Batch IDs stay
distinct whatever you pass, since every optimization in the program draws them
from one counter.

[examples/multiple_optimizations.py](https://github.com/TNO-ropt/ropt/blob/main/examples/multiple_optimizations.py)
runs one optimization per start vector, at most two at a time, and tags each with
its own metadata:

```python
--8<-- "examples/multiple_optimizations.py:run"
```

## Watching optimizations that overlap

For optimizations running on threads, `result_callback=` can receive a different
callback for each optimization, or the same callback for all of them. A shared
callback can be called concurrently, so it needs its own lock if it changes
shared data or writes to a shared file.

Alternatively, pass a shared handler in `handlers=`. A handler takes a lock
around each event it handles, so the optimizations wait for it one at a time.
Slow work in a shared handler therefore delays every optimization using it.
See [Sharing a handler across concurrent
optimizations](../results/handlers.md#sharing-a-handler-across-concurrent-runs)
for an example and details.

Optimizer output capture supports only one optimization at a time in each
process. If concurrent optimizations in the same process set
[`stdout` or `stderr`](../optimizer_setup/configuration_sections.md#optimizer),
the second to start raises [`WorkflowError`][ropt.exceptions.WorkflowError].
Leave both unset for optimizations running on threads, and set
[`verbose=False`](../optimizer_setup/configuration_sections.md#backend) to
avoid interleaved optimizer reports on the terminal.

## Failure in one optimization { #failure-in-one-run }

When an optimization in a `run` call raises an exception, `run` aborts the
others and raises that exception without returning their results. See
[Failures and Aborts](failures.md) for exception propagation, abort timing
and exit codes.

A callable that catches the exception of its optimization does not fail, so the
other optimizations are not aborted. Catch the exceptions you expect, and return
something that marks the failure:

```python
def optimize_or_none(x0):
    try:
        return optimize(config, x0, objective, pool=pool)
    except MySimulationError:
        return None

optimizations = [partial(optimize_or_none, x0) for x0 in start_points]
results = run(optimizations, pool=runner)
```

`MySimulationError` stands for an exception that your own objective raises.

## Aborting from outside { #stopping-from-outside }

A result callback can stop its own optimization. To stop work from outside
the optimization, for example from a user-interface thread, call
[`Session.abort`][ropt.Session.abort]. This aborts every call given one of
the session's pools or made in its `with` block. Each affected call raises
[`AbortedError`][ropt.exceptions.AbortedError] with `USER_ABORT` and returns
no result.

`Session.abort()` aborts the whole session, not just one optimization, and
the affected calls return no result. To retain intermediate results, collect
them as they arrive with a result callback or a [handler](../results/handlers.md).
To stop only one optimization when a criterion is met, use its result callback
instead. This stops only the optimization that owns the callback and returns
its best result; other optimizations in the session continue.

After `Session.abort()`, the session cannot be reused; open a new session to
start more work. See [Which calls are aborted](failures.md#which-calls-are-aborted) for
abort propagation and [What happens to running work](failures.md#what-an-abort-does)
for the effect on running evaluations.

### Ctrl-C and session aborts

Normally, Ctrl-C raises `KeyboardInterrupt`; it does not call `Session.abort()`.
See [Failures and Aborts](failures.md#what-a-failure-aborts) for which calls
are aborted when that exception propagates.

If you want Ctrl-C to abort the whole session explicitly instead, install a
`SIGINT` handler as below. This replaces the normal `KeyboardInterrupt`
behavior: the optimization raises `AbortedError` with `USER_ABORT`, and the
session cannot be reused.

```python
with session() as s:
    pool = s.thread_pool(workers=4)
    signal.signal(signal.SIGINT, lambda *_: s.abort())
    try:
        result = optimize(config, x0, objective, pool=pool)
    except AbortedError as error:
        print("aborted:", error.exit_code.message)
```

See [Keyboard Interrupts](../troubleshooting/keyboard_interrupt.md)
if Ctrl-C does not interrupt your program while it waits.
