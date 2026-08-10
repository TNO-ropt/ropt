# Running in Parallel

An optimization calls your objective function many times. By default these calls
happen one after another, on the same thread that called
[`optimize`][ropt.simple.optimize]. If each call is slow, you can run several at
the same time by opening an **execution block** first.

There are three kinds of block. You pick one with a `with` statement, and every
optimization inside it uses that block:

```python
from ropt.simple import optimize, threads

with threads(workers=4):
    result = optimize(config, x0, objective)
```

The block fixes one worker pool for everything inside it. You cannot open a
second block inside the first one.

## The three choices

### `threads` — a pool of threads in the same process

```python
from ropt.simple import threads
```

The evaluations run on background **threads**, all inside your program. Nothing
is copied between them, so any Python function works as the objective, and it can
freely use the data around it.

Use `threads` when each evaluation spends most of its time **waiting** — for
example when it starts an external program, reads a file, or calls a network
service. While one evaluation waits, the others can run.

Threads share one Python interpreter. Because of this, threads do **not** speed
up work that is pure Python number-crunching. For that, use `processes`.

### `processes` — a pool of separate processes

```python
from ropt.simple import processes
```

The evaluations run in **separate processes**. This gives real parallel speed for
heavy Python computations, because each process has its own interpreter.

Your objective is sent to the worker processes, so this needs the `cloudpickle`
extra (`pip install "ropt[cloudpickle]"`).

### `hpc` — a pool of jobs on a cluster

```python
from ropt.simple import hpc
```

The evaluations are submitted as jobs to an HPC queue (for example Slurm). Use
this when a single evaluation is a large job that belongs on a cluster. It needs
the `ropt[hpc]` extra and a reachable cluster:

```python
with hpc(workers=10):
    result = optimize(config, x0, objective)
```

With no further arguments, `hpc` uses the default cluster and queue from the
`pysqa` configuration of your `ropt` installation. Cluster-specific parameters —
such as the `cluster` name, the `queue`, and the number of `cores` per job — can
be passed to `hpc` when you need them; see
[Running Optimizations](../running/running.md#running-on-an-hpc-cluster) for the full list.

Like `processes`, work is sent to the cluster, so the `cloudpickle` extra is
needed here too (it is already part of the `ropt[hpc]` extra).

## Which one should I use?

| Situation | Use |
| --- | --- |
| Evaluations are fast | no block (the default) |
| Each evaluation mostly waits (external tool, I/O) | `threads` |
| Each evaluation is heavy Python computation | `processes` |
| Each evaluation is a big job for a cluster | `hpc` |

## Running multiple optimizations in parallel

The same blocks also power [`optimize_many`][ropt.simple.optimize_many], which
runs several optimizations at once:

```python
from ropt.simple import optimize_many, threads

with threads(workers=4):
    results = optimize_many(config, start_points, objective)
```

`optimize_many` always runs the optimizations themselves concurrently, each on
its own thread — that part does not depend on the block. The block instead
decides how the *function evaluations* inside those runs are parallelized: they
share its one worker pool. So `threads(workers=1)` runs the optimizations
together but evaluates one point at a time, while a larger pool (or `processes`
or `hpc`) evaluates several at once. An execution block is required.

See [Running Optimizations](../running/running.md#many-optimizations-at-once) for more.
