# Running in Parallel

An optimization calls your evaluation function many times. By default these calls
happen one after another, on the same thread that called
[`optimize`][ropt.simple.optimize]. If each call is slow, you can run several at
the same time by evaluating on a **worker pool**.

Pools come from a [`session`][ropt.simple.session]. Open one, ask it for the kind
of pool you want, and pass that pool to the run:

```python
from ropt.simple import optimize, session

with session() as s:
    result = optimize(config, x0, objective, pool=s.thread_pool(workers=4))
```

There are three kinds of pool. A session can hand out as many as you like, of any
kind, and each run uses the one you give it — and only that one. Closing the
session releases them all.

Where your objective runs depends on which pool you pass (or none):

```mermaid
flowchart TB
    subgraph proc["your Python process"]
        main(["your program<br/>(main thread)"])
        seq["no pool —<br/>one eval at a time"]
        th["thread_pool —<br/>worker threads<br/>(share memory)"]
    end
    wp["process_pool —<br/>separate processes<br/>(data copied)"]
    clu["hpc_pool —<br/>cluster jobs<br/>(data copied)"]
    main --> seq
    main --> th
    main --> wp
    main --> clu
```

Threads stay **inside** your Python process and share its memory, so any Python
function works and nothing is copied. A process or HPC pool runs the work
**outside** your process, so the objective and its data are copied there.

??? info "New to threads and processes?"
    A **process** is a running program with its own private memory. A **thread**
    is a worker inside a process, and all threads in a process share that memory.
    Threads are cheap and share data for free, but Python runs only one thread's
    code at a time, so threads do not speed up pure computation — only work that
    waits (for a file, a network reply, an external tool). Separate **processes**
    each have their own interpreter and run truly in parallel, but they do not
    share memory, so data has to be copied between them.

## The three choices

### `thread_pool` — a pool of threads in the same process

```python
pool = s.thread_pool(workers=4)
```

The evaluations run on background **threads**, all inside your program. Nothing
is copied between them, so any Python function works as the objective, and it can
freely use the data around it.

Use a thread pool when each evaluation spends most of its time **waiting** — for
example when it starts an external program, reads a file, or calls a network
service. While one evaluation waits, the others can run.

Threads share one Python interpreter. Because of this, threads do **not** speed
up work that is pure Python number-crunching. For that, use a process pool.

### `process_pool` — a pool of separate processes

```python
pool = s.process_pool(workers=4)
```

The evaluations run in **separate processes**. This gives real parallel speed for
heavy Python computations, because each process has its own interpreter.

Your objective and its data are **copied** to the worker processes, so they must
be serializable: an objective defined at module level works out of the box, while
a lambda (a one-line, unnamed function), a closure (a function defined inside
another function), or a function defined in a notebook cell needs the
`cloudpickle` extra (see
[Installation](installation.md#optional-extras)).

Because each worker is a separate process, your objective can only send results
**back** through its return value; it cannot share memory with your handlers or
the rest of your program. See
[Handlers and the process boundary](../running/handlers.md#handlers-and-the-process-boundary).

### `hpc_pool` — a pool of jobs on a cluster

```python
pool = s.hpc_pool(workers=10)
```

The evaluations are submitted as jobs to an HPC queue (for example Slurm). Use
this when a single evaluation is a large job that belongs on a cluster. It needs
the `ropt[hpc]` extra and a reachable cluster:

```python
with session() as s:
    result = optimize(config, x0, objective, pool=s.hpc_pool(workers=10))
```

With no further arguments, `hpc_pool` uses the default cluster and queue from the
`pysqa` configuration of your `ropt` installation — `pysqa` is the package `ropt`
uses to submit and track cluster jobs. Cluster-specific parameters —
such as the `cluster` name, the `queue`, and the number of `cores` per job — can
be passed to `hpc_pool` when you need them; see
[Parallel Execution and Many Runs](../running/parallel.md#running-on-an-hpc-cluster) for the full list.

Like a process pool, the work is copied to the cluster, so the same rule about
which functions can be sent applies; add `ropt[cloudpickle]` to lift it (see
[Installation](installation.md#optional-extras)).

## Which one should I use?

| Pool | Where evaluations run | Data | Speeds up heavy Python? | Use when |
| --- | --- | --- | --- | --- |
| none (default) | the calling thread, one at a time | shared | no | evaluations are fast |
| `thread_pool` | background threads, one process | shared | no — one interpreter | each evaluation mostly **waits** (external tool, I/O) |
| `process_pool` | separate processes | copied | yes | each evaluation is heavy **Python computation** |
| `hpc_pool` | jobs on a cluster | copied | yes | each evaluation is a big **cluster job** |

## Running multiple optimizations in parallel

The same pools also power [`optimize_many`][ropt.simple.optimize_many], which
runs several optimizations at once:

```python
from ropt.simple import optimize_many, session

with session() as s:
    pool = s.thread_pool(workers=4)
    results = optimize_many(config, start_points, objective, pool=pool)
```

`optimize_many` always runs the optimizations themselves concurrently, each on
its own thread — that part does not depend on the pool. The pool instead decides
how the *function evaluations* inside those runs are parallelized: they all
happen on it. So `thread_pool(workers=1)` runs the optimizations together but
evaluates one point at a time, while a larger pool (or a process or HPC pool)
evaluates several at once. Without a pool the runs evaluate on their own driver
threads, so your objective is then called by several threads at once.

### Collecting results from concurrent runs

[Collecting Results with Handlers](handlers.md) showed a handler reused across
a **sequential** loop, accumulating one run's results after another. That does
not work here: the runs of `optimize_many` overlap in time, and a plain
handler cannot safely collect from several runs at once. Instead, build a
**shared handler group** on the session with `shared_handlers`, and pass the
group where you would pass the handler:

```python
from ropt.simple import HistoryHandler, optimize_many, session

history = HistoryHandler()
with session() as s:
    pool = s.thread_pool(workers=4)
    collected = s.shared_handlers(history)
    results = optimize_many(
        config, start_points, objective, pool=pool, handlers=[collected]
    )

print(history.results)   # every result, from every run, safely collected
```

`optimize_many` only accepts groups in `handlers=` — a bare handler is rejected
there, because its runs overlap.

See [Parallel Execution and Many Runs](../running/parallel.md#many-optimizations-at-once) for more.

## Offloading your own functions

A pool can also run **your own** functions in parallel, not just the optimizer's
evaluations. Pass a function — or several — to [`offload`][ropt.simple.offload],
along with the pool you want to run it on:

```python
from functools import partial

from ropt.simple import offload, session

with session() as s:
    result = offload(partial(expensive, data), pool=s.process_pool(workers=4))
```

This is for expensive, self-contained work in code you write — a helper, a custom
step, or a transform. Like the objective, such functions are copied to the
workers on a process or HPC pool. Without a pool, `offload` simply runs the
function inline instead, so your code never needs a separate fallback for the
case where no pool is available. See
[Parallel Execution and Many Runs](../running/parallel.md#offloading-your-own-work) for
details.
