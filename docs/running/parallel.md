# Parallel Execution and Many Runs

This page continues [Running Optimizations](running.md): running evaluations
in parallel, running several optimizations at once, and offloading your own
work to a pool.

## Running in parallel

By default `optimize` runs on the calling thread, one evaluation at a time. To
run the evaluations in parallel, open a [`session`][ropt.simple.session], ask it
for a **pool**, and pass that pool to the run. See
[Running in Parallel](../getting_started/execution.md) for a full explanation of the three choices
and their trade-offs:

```python
from ropt.simple import session

with session() as s:
    result = optimize(config, x0, objective, pool=s.thread_pool(workers=8))
```

A session is a background event loop that the pools run on. Closing it releases
every pool it created, so that is normally all the cleanup you need. Nothing is
implicit: a run evaluates on the pool you hand it, and on no other. A run given
no pool evaluates in-process, wherever it is called from — including from a
thread you started yourself.

!!! tip "How a batch is split across workers"
    Each evaluation in a batch is transferred to a worker as its own task by
    default, which spreads the batch as widely as the pool allows. Every
    transfer costs something, though, so when the evaluations are cheap the
    transfers can dominate. Set `bundle_size=` on the pool to group several
    evaluations into one task, or `bundle_size=0` to send the whole batch as a
    single task. The evaluations within a task run one after another, so `0`
    gives up parallelism inside the batch entirely: it is for a pool whose
    parallelism comes from the runs above it, as in the note below.

You can keep several pools open at once and choose per run:

```python
with session() as s:
    fast = s.thread_pool(workers=8)
    heavy = s.process_pool(workers=4)
    cheap = optimize(config, x0, objective, pool=fast)
    costly = optimize(config, x0, expensive_objective, pool=heavy)
```

!!! note "Pools inside an evaluation"
    An evaluation function may start a run of its own and give it a pool, on two
    conditions.

    It must be a **different** pool. A nested run waits for its own evaluations
    to finish, so one handed the pool it is already running on would wait for
    the workers it is itself occupying — a deadlock as soon as they are all
    busy, which is the normal case, since a run fills its pool with one work
    item per realization. Rather than hang, the pool refuses work submitted by
    the evaluation itself with a
    [`WorkflowError`][ropt.exceptions.WorkflowError]. A thread the evaluation
    starts is on its own: it is not recognized as a worker, so it can still
    deadlock on the pool. Give the inner run its own pool, or a
    [`serial_pool`][ropt.simple.serial_pool], which evaluates inline and can
    always be reused.

    The evaluation must stay **in your process**, so on a thread pool or a
    serial pool. On a process or HPC pool the evaluation function is copied into
    a worker, and a pool cannot be copied with it: build the inner pool inside
    the worker, from a session opened there, or run the inner optimization
    without one. An evaluation function that carries a pool along anyway is
    stopped in the worker, which names what it was handed instead of failing
    somewhere deep inside the run.

!!! tip "Releasing a pool early"
    A pool holds its workers until the session closes. That is usually fine, but
    if you build pools in a loop inside one long-lived session — above all
    process pools, which hold worker interpreters — release each one when you
    are done with it, either with `pool.close()` or by using it as a context
    manager:

    ```python
    with session() as s:
        for case in cases:
            with s.process_pool(workers=4) as pool:
                optimize(config, case, objective, pool=pool)
    ```

    A closed pool cannot be reopened, and a run still using it stops with
    [`ExecutorStopped`][ropt.exceptions.ExecutorStopped]. Starting a *new* run
    on it is refused before anything runs, with a
    [`WorkflowError`][ropt.exceptions.WorkflowError] saying the pool is closed —
    which is what you get if a pool outlives the `with session()` block that
    created it.

### Running on an HPC cluster

An [`hpc_pool`][ropt.simple.Session.hpc_pool] submits each evaluation as a job to
an HPC queue (through `pysqa`); it needs the `ropt[hpc]` extra. With no further
arguments it uses the default cluster and queue from the `pysqa` configuration of
your `ropt` installation:

```python
from ropt.simple import session

with session() as s:
    result = optimize(config, x0, objective, pool=s.hpc_pool(workers=10))
```

`hpc_pool` accepts the following parameters:

| Parameter     | Description                                                                |
| ------------- | ------------------------------------------------------------------------- |
| `workers`     | Maximum number of concurrent cluster jobs (default: 1).                   |
| `cores`       | Number of CPUs per job (default: 1).                                      |
| `cluster`     | Cluster name, when the `pysqa` config defines several.                    |
| `queue`       | Queue or partition name.                                                  |
| `workdir`     | Shared-filesystem working directory (defaults to the current directory).  |
| `config_path` | Path to the `pysqa` configuration directory.                              |
| `template`    | Inline submission-script template, used instead of a config.              |
| `queue_type`  | Queueing system type (default: `"slurm"`).                                |
| `bundle_size` | Evaluations bundled into one cluster job, `0` for the whole batch as one job (default: 1). See [How a batch is split across workers](#running-in-parallel) above. |

### Evaluating in-process, on purpose

[`serial_pool`][ropt.simple.serial_pool] is a pool with no workers: it carries
only the batch-ID counter that the runs sharing it draw from, and their
evaluations happen in-process on the calling thread. It needs no session, and
needs no releasing.

Use it to give several runs one continuous batch-ID sequence without running
their evaluations in parallel, or simply to say in the code that a run is meant
to evaluate in-process.

## Many optimizations at once

To run several optimizations together, use
[`optimize_many`][ropt.simple.optimize_many]. Any of `config`, `x0`, or
`objective` may be a single value (used for every run) or a list (one per run):

```python
from ropt.simple import optimize_many, session

with session() as s:
    pool = s.thread_pool(workers=4)
    results = optimize_many(config, start_points, objective, pool=pool)  # one run per start
```

!!! tip "Give each run an ID"
    Pass a per-run `metadata` list to tag every run with a user-defined
    identifier that travels with its results (and shows up in a
    [`DataFrameHandler`](handlers.md#dataframehandler)'s tables):

    ```python
    labels = ["low", "mid", "high"]
    results = optimize_many(
        config, start_points, objective, metadata=[{"run_id": x} for x in labels]
    )
    for result in results:
        print(result.results.metadata["run_id"])
    ```

    See [Attaching metadata](running.md#attaching-metadata) for details.

There are two independent levels of concurrency here:

- **The optimizations** always run concurrently, each on its own driver thread.
  This is built into `optimize_many` and does not depend on the pool;
  the `limit` argument caps how many run at the same time.
- **The function evaluations** inside those runs all happen on the one pool you
  pass, and the pool decides how they are parallelized. With
  `thread_pool(workers=1)` the runs still progress together, but their
  evaluations are executed one at a time. A larger pool — `thread_pool(workers=n)`,
  `process_pool`, or `hpc_pool` — runs several evaluations at once.

Sharing one pool is also what keeps the runs' batch IDs apart, since they draw
from its single counter.

The two callback arguments differ in the same way. `report=` is **per run**: one
callback watches every run, or pass a list with one callback per run. `handlers=`
is **shared**: one list of groups that all runs feed together, which is why a
plain handler is refused there — see [Sharing a handler across concurrent
runs](handlers.md#sharing-a-handler-across-concurrent-runs).

!!! warning "Without a pool the driver threads do the evaluating"
    `optimize_many` needs no session and no pool. Without one, the runs still
    execute concurrently, but each evaluates in-process on its own driver
    thread — so your evaluation function is called by several threads at once
    and must tolerate that. Give the call a pool, or a
    [`serial_pool`][ropt.simple.serial_pool] if you want one shared batch-ID
    sequence, when it must not be.

## Offloading your own work

You can hand **your own** functions to a pool with
[`offload`][ropt.simple.offload]. It is useful when code you control — a custom
step, a domain transform, or a helper you call between optimizations — has an
expensive, self-contained piece of work you want to run on a pool instead of
inline.

Pass a single callable to run one call and get its result back:

```python
from functools import partial

from ropt.simple import offload, session

with session() as s:
    result = offload(partial(expensive, data), pool=s.process_pool(workers=4))
```

`offload` takes **zero-argument** callables — bind arguments with
`functools.partial` (or a closure). Pass a **sequence** of callables to run them
concurrently and get a tuple of results in order; they may be entirely different
functions:

```python
with session() as s:
    pool = s.process_pool(workers=4)
    first, second = offload([partial(expensive, x), partial(other, y)], pool=pool)
```

As with the evaluation function on a process or HPC pool, the callables and their
arguments are **copied to the workers**, since they run in separate processes.

### Without a pool

`offload` with no pool — or with a [`serial_pool`][ropt.simple.serial_pool] —
runs the callables inline, on the calling thread. So code that may or may not
have a pool to hand needs no guard and no fallback: pass along whatever it has,
including `None`.

```python
def transform(x, pool=None):
    return offload(partial(expensive, x), pool=pool)
```

!!! note "Not from an inline handler in a shared group"
    A handler in a [shared group](handlers.md#sharing-a-handler-across-concurrent-runs)
    that runs inline runs on the session's event loop; offloading to a pool on
    that same session would starve the very loop it is waiting on, so it raises
    a [`WorkflowError`][ropt.exceptions.WorkflowError]. A
    [`threaded`](handlers.md#running-a-handler-in-a-thread) handler runs on a dispatcher
    worker instead and can offload, as can a local handler, which runs on the
    thread driving the run.

    Better still, do parallel work from your optimization code and leave
    handlers to handle results.

## Where to next

- Collect or react to every result, not just the best one:
  [Result Handlers](handlers.md).
- Back to the basics of a single run: [Running Optimizations](running.md).
