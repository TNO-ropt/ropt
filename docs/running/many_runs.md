# Several Optimizations at Once

Several optimizations can run together rather than one after another. Wrap each
`optimize` call in a zero-argument callable and pass the callables to
[`Session.offload`][ropt.Session.offload], which runs each one on a thread of its
own and returns the results in the order the callables were given:

```python
from functools import partial

from ropt import session

with session() as s:
    pool = s.thread_pool(workers=4)
    results = s.offload(
        [partial(pool.optimize, config, x0, objective) for x0 in start_points]
    )
```

Each callable is an ordinary `optimize` call, so every optimization can have its
own configuration, initial values, objective, `metadata` and `report` callback.

!!! tip "Give each optimization an ID"
    Pass `metadata` to tag each optimization with an identifier that is
    attached to its results (and shows up in a
    [`DataFrameHandler`](../results/handlers.md#dataframehandler)'s tables):

    ```python
    labels = ["low", "mid", "high"]
    results = s.offload(
        [
            partial(pool.optimize, config, x0, objective, metadata={"label": label})
            for x0, label in zip(start_points, labels)
        ]
    )
    for result in results:
        print(result.results.metadata["label"])
    ```

    See [Attaching metadata](running.md#attaching-metadata) for details.

## Two levels of concurrency

There are two independent levels of concurrency here:

- **The optimizations** run concurrently, each on a thread that `offload` starts
  for it. The `limit` argument of `offload` caps how many run at the same time;
  without it, all of them start at once.
- **The function evaluations** inside those optimizations happen on the pool
  that each `optimize` call is made on, and the pool determines how they are
  parallelized. With `thread_pool(workers=1)` the optimizations still progress
  together, but their evaluations are executed one at a time. A larger pool —
  `thread_pool(workers=n)`, or a process, local or HPC pool — runs several
  evaluations at once.

One pool is one budget: `workers=10` means ten evaluations at a time across all
the optimizations that use the pool, not ten per optimization. Batch IDs stay
distinct whatever you pass, since every optimization in the program draws them
from one counter.

[examples/several_optimizations.py](https://github.com/TNO-ropt/ropt/blob/main/examples/several_optimizations.py)
runs one optimization per start vector, at most two at a time, and tags each with
its own metadata:

```python
--8<-- "examples/several_optimizations.py:run"
```

## Watching optimizations that overlap

`report=` belongs to one `optimize` call: give each optimization its own
callback, or the same one to all of them. `handlers=` takes handlers that
several optimizations may feed together — see [Sharing a handler across
concurrent optimizations](../results/handlers.md#sharing-a-handler-across-concurrent-runs).

!!! warning "A shared `report=` callback is called by every optimization at once"
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

!!! warning "Without a pool, the threads of `offload` do the evaluating"
    An optimization started with [`Session.optimize`][ropt.Session.optimize]
    evaluates in your own program, on the thread that `offload` started for it.
    When several of them run at once, your evaluation function is called by
    several threads at once and must tolerate that. Start them on a pool when it
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

The optimizations started from the callables of one `offload` are
[nested](failures.md#nested-tasks) in it. When one of them raises, `offload`
aborts the others and raises that exception at once; when `offload` was called
from your script, every other task on the session is aborted as well.
[Failures and Aborts](failures.md) gives the rules for every case. An aborted
optimization raises `AbortedError` with `ABORTED_ON_ERROR` — at once on a pool,
or at its next evaluation boundary without one — but `offload` does not wait for
it, and its result is lost. Optimizations still queued behind `limit` are not
started after the abort.

A callable that catches the exception of its optimization does not fail, so the
other optimizations are not aborted. Catch the exceptions you expect, and return
something that marks the failure:

```python
def optimize_or_none(x0):
    try:
        return pool.optimize(config, x0, objective)
    except SimulationError:
        return None

results = s.offload([partial(optimize_or_none, x0) for x0 in start_points])
```

`SimulationError` stands for an exception that your own objective raises.

[`Session.abort`](running.md#stopping-from-outside) reaches every optimization on
the session, and `offload` then raises
[`AbortedError`][ropt.exceptions.AbortedError] with `USER_ABORT`. When a failure
elsewhere aborted it, the exit code is `ABORTED_ON_ERROR`, so the exit code
distinguishes an abort that was asked for from one that a failure caused. A
`KeyboardInterrupt` aborts the optimizations of the `offload` it passes through,
and is raised at once. See [Exit Codes](../results/exit_codes.md) for what each
code means.
