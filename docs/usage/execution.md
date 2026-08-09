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

There is one rule to keep in mind: everything that crosses into a worker process
must be **picklable** (it is sent by copying). In practice this means:

- your objective must be a **module-level function** (not a lambda, and not a
  function defined inside another function);
- the data it uses must also be picklable (plain arrays, numbers, and so on).

If you see a pickling error, this rule is usually the cause.

### `hpc` — a pool of jobs on a cluster

```python
from ropt.simple import hpc
```

The evaluations are submitted as jobs to an HPC queue (for example Slurm). Use
this when a single evaluation is a large job that belongs on a cluster. It needs
the `ropt[hpc]` extra and a reachable cluster:

```python
with hpc(workers=10, cluster="slurm"):
    result = optimize(config, x0, objective)
```

Like `processes`, work is sent to the cluster, so the same picklability rule
applies.

## Which one should I use?

| Situation | Use |
| --- | --- |
| Evaluations are fast | no block (the default) |
| Each evaluation mostly waits (external tool, I/O) | `threads` |
| Each evaluation is heavy Python computation | `processes` |
| Each evaluation is a big job for a cluster | `hpc` |

## Running many optimizations together

The same blocks also power [`optimize_many`][ropt.simple.optimize_many], which
runs several optimizations at once and shares the one worker pool between them:

```python
from ropt.simple import optimize_many, threads

with threads(workers=4):
    results = optimize_many(config, start_points, objective)
```

See [The Simple API](simple.md#many-optimizations-at-once) for more.
