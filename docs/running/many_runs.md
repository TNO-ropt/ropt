# Running Multiple Optimizations

Several optimizations can run together rather than one after another. Wrap each
`optimize` call in a zero-argument callable and pass the callables to
[`run`][ropt.run] with a thread pool. `run` starts each one on a worker of that
pool and returns the results in the order the callables were given:

```python
from functools import partial

from ropt import optimize, run, session

with session() as s:
    pool = s.thread_pool(workers=4)
    runner = s.thread_pool(workers=len(start_points))
    results = run(
        [partial(optimize, config, x0, objective, pool=pool) for x0 in start_points],
        pool=runner,
    )
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

`result_callback=` belongs to one `optimize` call: give each optimization its own
callback, or the same one to all of them. `handlers=` takes handlers that
several optimizations may feed together — see [Sharing a handler across
concurrent optimizations](../results/handlers.md#sharing-a-handler-across-concurrent-runs).

!!! warning "A shared result callback is called by every optimization at once"
    Each optimization calls its callback on its own thread. Nothing serializes
    those calls, so a callback that appends to a list, updates a counter, or
    writes a file needs a lock of its own. Give each optimization its own
    callback when they must stay apart, or pass a
    [handler](../results/handlers.md#sharing-a-handler-across-concurrent-runs) in
    `handlers=`, which takes a lock around every event it handles.

!!! warning "A shared handler makes the optimizations wait for each other"
    That lock is not free. An optimization that emits a result waits until the
    handler has finished with it, and a second optimization waits for the first.
    A slow handler therefore throttles all of them, once per result produced.

    So keep a shared handler cheap. A handler that must do slow work — writing
    a file, talking to a database — holds up every optimization feeding it.

!!! warning "Without a pool, the workers of `run` do the evaluating"
    An optimization started with [`optimize`][ropt.optimize] without a pool
    evaluates in your own program, on the worker of `run` that runs it.
    When several of them run at once, your evaluation function is called by
    several threads at once and must tolerate that. Give them a pool when it
    must not be.

!!! warning "Not every backend can take part"
    An optimizer that needs a working directory of its own, writes to a file
    whose name is fixed, or keeps state inside its library between calls cannot
    run while anything else is running in the same process — another
    optimization of its own kind included. Each backend documents whether this
    applies to it. Select it as
    [`external/...`](../optimizer_setup/optimizer.md#external-backend) and
    it gets a process of its own, where none of that is shared.

    Optimizer output capture is likewise for one optimization at a time. If more
    than one of them sets
    [`stdout` or `stderr`](../optimizer_setup/configuration_sections.md#optimizer), the
    second to start raises [`WorkflowError`][ropt.exceptions.WorkflowError].
    Leave both unset here, and set [`verbose=False`](../optimizer_setup/configuration_sections.md#backend)
    unless you want the optimizers' reports interleaved on the terminal.

## Failure in one optimization { #failure-in-one-run }

The optimizations started from the callables of one `run` call are
[nested](failures.md#nested-calls) in it. When one of them raises, `run`
aborts the others and raises that exception at once.
[Failures and Aborts](failures.md) gives the rules for every case. An aborted
optimization raises `AbortedError` with `ABORTED_ON_ERROR` — at once with a pool,
or at its next evaluation boundary without one — but `run` does not wait for
it, and its result is lost. Optimizations still waiting for a worker are not
started after the abort.

A callable that catches the exception of its optimization does not fail, so the
other optimizations are not aborted. Catch the exceptions you expect, and return
something that marks the failure:

```python
def optimize_or_none(x0):
    try:
        return optimize(config, x0, objective, pool=pool)
    except SimulationError:
        return None

results = run([partial(optimize_or_none, x0) for x0 in start_points], pool=runner)
```

`SimulationError` stands for an exception that your own objective raises.

[`Session.abort`](#stopping-from-outside) reaches every optimization on
the session's pools, and `run` then raises
[`AbortedError`][ropt.exceptions.AbortedError] with `USER_ABORT`. When a failure
aborted it, the exit code is `ABORTED_ON_ERROR`, so the exit code
distinguishes an abort that was asked for from one that a failure caused. A
`KeyboardInterrupt` aborts the optimizations of the `run` call it passes through,
and is raised at once. See [Exit Codes](../results/exit_codes.md) for what each
code means.

## Aborting from outside { #stopping-from-outside }

The result callback runs inside the optimization it stops, which is no use to
a signal handler or a user interface. [`Session.abort`][ropt.Session.abort] is
the one that is called from another thread: it aborts every call given one of
the session's pools or made in its `with` block, and each raises
[`AbortedError`][ropt.exceptions.AbortedError] with `USER_ABORT`.

```python
with session() as s:
    pool = s.thread_pool(workers=4)
    signal.signal(signal.SIGINT, lambda *_: s.abort())
    try:
        result = optimize(config, x0, objective, pool=pool)
    except AbortedError as error:
        print("aborted:", error.exit_code.message)
```

The two are not the same kind of ending, which is why they have different exit
codes. A result callback stops the optimization *on a criterion*, at a point it
chose, and the optimization returns the best result it had reached. An abort
applies no criterion, and the optimization returns no result. To keep what an
aborted optimization had reached, collect its results as they arrive, with a
result callback or a [handler](../results/handlers.md).

An aborted optimization does not wait for the evaluations that are running.
With no pool, the optimization runs the evaluation function itself, so the
evaluation in progress returns first. On a thread or process pool, the
evaluations already running continue after the optimization has ended, and their workers are free only once they
return. With an HPC pool they are cancelled.

An abort is permanent: the session stays aborted. A call given one of its
pools afterwards, or in its `with` block, raises `AbortedError` with `USER_ABORT` at
once. Code that catches the `AbortedError` inside the block can record what
happened, but cannot start more work in the session; catching it outside the
`with` block works as well. To try again, open a new session. A
[`run`][ropt.run] call is aborted as a whole: a callable still waiting for a
worker is not started. Leaving the session's `with` block
aborts its calls as well, and then releases its pools, which is what refuses a
call made after that. Releasing an HPC pool cancels the jobs it still
has, so none are left running when the block has ended. A call aborted that way
raises with `ABORTED` rather than `USER_ABORT`, since it was the block ending
rather than a request to abort.

A call without a pool, such as [`optimize`][ropt.optimize], made in the
`with` block of a session, is aborted with the session. Made outside any
session block, it cannot be aborted this way.
