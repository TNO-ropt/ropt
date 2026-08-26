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

There are four kinds of pool. Evaluating in place — with no pool, or with an
explicit [`serial_pool`][ropt.simple.serial_pool] — is a fifth choice rather
than the absence of one. A session can hand out as many pools as you like, of
any kind, and each run uses the one you give it — and only that one. Closing the
session releases them all.

Where your objective runs depends on which pool you pass (or none):

```mermaid
flowchart TB
    subgraph proc["your Python process"]
        main(["your program<br/>(main thread)"])
        seq["no pool / serial_pool —<br/>one eval at a time"]
        th["thread_pool —<br/>worker threads<br/>(share memory)"]
    end
    wp["process_pool —<br/>a few reused processes<br/>(data copied)"]
    loc["local_pool —<br/>one process per eval<br/>(data copied)"]
    clu["hpc_pool —<br/>cluster jobs<br/>(data copied)"]
    main --> seq
    main --> th
    main --> wp
    main --> loc
    main --> clu
```

Threads stay **inside** your Python process and share its memory, so any Python
function works and nothing is copied. The other three run the work **outside**
your process, so the objective and its data are copied there.

??? info "New to threads and processes?"
    A **process** is a running program with its own private memory. A **thread**
    is a worker inside a process, and all threads in a process share that memory.
    Threads are cheap and share data for free, but Python runs only one thread's
    *Python* code at a time. Work that **waits** — for a file, a network reply,
    an external tool — overlaps freely, because a waiting thread holds nothing;
    and so does work a library performs outside Python, as `numpy` and friends
    do while they crunch an array. What is stuck one-at-a-time is arithmetic
    written in Python itself. Separate **processes** each have their own
    interpreter and always run truly in parallel, but they do not share memory,
    so data has to be copied between them.

## The choices

### No pool, or `serial_pool` — evaluate in place

```python
from ropt.simple import optimize, serial_pool

result = optimize(config, x0, objective)                      # no pool
result = optimize(config, x0, objective, pool=serial_pool())  # the same, said out loud
```

The evaluations happen one after another on the thread that called `optimize`.
This is the default, and for a fast objective it is also the right answer: a
pool costs something to set up and to hand work to, and below a certain
evaluation cost that is all it does.

[`serial_pool`][ropt.simple.serial_pool] is that same behaviour named
explicitly. It needs no session and holds no workers, and it is worth passing
when running in-process is a decision rather than an oversight — or when several
runs should share one batch-ID sequence.

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

Threads share one Python interpreter, so arithmetic written in Python itself does
not get faster on more threads. Array libraries are a different matter: `numpy`
and its kin do their work outside Python and let the other threads run
meanwhile. "My objective computes" is therefore not on its own a reason to reach
past this pool — see [Which one should I use?](#which-one-should-i-use).

### `process_pool` — a pool of separate processes

```python
pool = s.process_pool(workers=4)
```

The evaluations run in **separate processes**. This gives real parallel speed for
heavy Python computations, because each process has its own interpreter.

Reach for it when the computation is **Python code**, or when each evaluation
needs its own copy of something a library keeps globally. An objective that
mostly runs an external program gains nothing here that a thread pool would not
have given more cheaply.

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

!!! warning "This pool does not clean up programs your objective started"

    A process pool reuses a handful of worker processes. When a run is stopped —
    by Ctrl-C, or by closing the pool — those workers are killed, but anything
    they had launched themselves is not: a simulator or solver started by your
    objective keeps running, unattached, after your program is gone. Nothing
    warns you. If your objective launches external programs, use `local_pool`
    below, which was built for exactly this.

### `local_pool` — one process per evaluation, on this machine

```python
pool = s.local_pool(workers=4)
```

Each evaluation gets a **process of its own**, started fresh and thrown away
afterwards, with its output captured to a file. It sits between a process pool
and a cluster: no queueing system, nothing to install, no extras.

It is the right choice when an evaluation is a self-contained *job* rather than
a Python function call — it runs a simulator, writes files, and takes long
enough that starting a process for it is noise. Two things it gives that
`process_pool` does not:

- **Stopping actually stops.** The evaluation and everything it launched are
  signalled together, so an interrupted run does not leave simulators behind.
- **Output is kept.** Whatever a failed evaluation printed is captured, and its
  last lines are attached to the error you see.

Those files live in a temporary directory that the pool cleans up after itself —
unless an evaluation failed, in which case the directory stays, with that
evaluation's output in it, and `ropt` logs where it is. To choose the location
yourself, pass one; a directory you passed is yours, and is never removed:

```python
pool = s.local_pool(workers=4, workdir="/scratch/my-run")
```

This pool is **POSIX only** — on Windows, creating it fails rather than quietly
offering less. As with a process pool, the objective and its data are copied,
so the same rule about which functions can be sent applies.

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

This is `local_pool` with a scheduler in front of it: the same shape of one
process per evaluation, the same captured output, the same cancellation — only
the process now starts on a compute node instead of this machine. So an
objective that already works on a `local_pool` will work here, and the step up
to a cluster is a change of pool rather than a change of objective.

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

Start with the question that rules choices *out*, because it is the only one you
can answer by reading your own code rather than by measuring:

!!! question "Does your objective read or write anything outside its arguments and its return value?"

    Global variables, a cache, a logger, an open file or database handle, an
    event handler, a counter it increments — anything at all that outlives one
    call.

    - **No.** Every pool works. Choose on speed alone, and you can swap between
      them freely later.
    - **Yes.** Stay with threads or in-process. Those live in your program and
      see the same memory. `process_pool`, `local_pool` and `hpc_pool` run the
      objective somewhere else, on a *copy* of everything it touched — so the
      writes land in that copy and vanish, and the reads see whatever the copy
      was made from. Nothing raises; the numbers just come out wrong.

Then, on speed:

| Pool | Where evaluations run | Data | Speeds up heavy Python? | Use when |
| --- | --- | --- | --- | --- |
| none / `serial_pool` | the calling thread, one at a time | shared | no | evaluations are fast |
| `thread_pool` | background threads, one process | shared | no — one interpreter | each evaluation mostly **waits** (external tool, I/O), or spends its time in `numpy` |
| `process_pool` | a few reused processes | copied | yes | each evaluation is heavy **Python computation** |
| `local_pool` | one process per evaluation | copied | yes | each evaluation is a self-contained **job** on this machine |
| `hpc_pool` | jobs on a cluster | copied | yes | each evaluation is a big **cluster job** |

??? tip "How to decide, without guessing"
    There is no reliable rule for whether threads will scale on a given
    objective. "I use `numpy`" says almost nothing: whether the GIL is released
    depends on the operation, the dtype and the array size, and a real objective
    is a mixture whose Python-level share is invisible to the person who wrote
    it.

    What makes the answer cheap is an asymmetry: **threads are the cheap thing
    to try, processes are the expensive commitment.** Trying a thread pool costs
    one argument, and its failure mode is *no speedup* — not breakage. So:

    1. Start with `thread_pool`. Time `workers=1` against `workers=4` on a
       shortened run.
    2. If it scales, you are done, and you never needed to know what the GIL was
       doing.
    3. If it does not, set `OMP_NUM_THREADS=1` (see below) and time it again.
    4. Only then pay for processes.

    Directional guidance is fine as orientation — waiting on an external program
    almost always scales, arithmetic written in Python never does, array-heavy
    work depends — as long as you treat it as a place to start rather than an
    answer. The answer is the measurement.

??? tip "If more workers makes it *slower*"
    `numpy`, `scipy` and similar are already multi-threaded underneath, through
    a BLAS library that by default takes **every core on the machine**. Run four
    evaluations at once and you have four such libraries each doing that: the
    machine is oversubscribed several times over, the threads fight for cores,
    and everything slows down.

    The symptom is actively misleading, because it reads as "parallelism does
    not help here" and pushes people towards processes — where the identical
    problem is waiting one layer down.

    The fix is to give each evaluation one core's worth of library threads,
    before `numpy` is imported:

    ```bash
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    ```

    Then let the pool provide the parallelism instead.

## Stopping a run

Press Ctrl-C, or close the pool, and `ropt` stops handing out new work at once.
What happens to the evaluations already running depends on the pool, because
what *can* be done to them differs:

| Pool | Evaluations already running |
| --- | --- |
| none / `serial_pool` | the current one finishes |
| `thread_pool` | they **run to completion** — a thread cannot be interrupted |
| `process_pool` | the worker processes are **killed** (but see the warning above) |
| `local_pool` | each evaluation **and everything it launched** is killed |
| `hpc_pool` | the jobs are **deleted from the queue** |

Two consequences are worth knowing before you need them.

**A thread pool cannot be hurried.** Python provides no way to interrupt a
running thread from outside, so a long evaluation on a `thread_pool` decides for
itself when it ends, and your program cannot exit before it does. `ropt` says so
out loud — a warning naming how many evaluations it is waiting for — because
otherwise it is indistinguishable from a hang. If an evaluation may run long and
has to be interruptible, put it on one of the other pools.

**Stopping is a firm request, not a guarantee.** On the pools that kill,
everything is asked to end and not waited for. A program that ignores the
request, or that is stuck inside the operating system, outlives it. What you get
is an interrupted run that exits promptly instead of waiting out the current
batch, which is the point — but "stopped" does not mean "nothing of it is left".

!!! tip "If Ctrl-C seems to do nothing at all"
    A handful of third-party packages, when imported, change a process-wide
    setting that stops Ctrl-C from breaking into a program that is *waiting* —
    and it then affects your whole program, not just `ropt`. `ropt` leaves that
    setting alone, because it belongs to your program rather than to a library
    it happens to use.

    You do not need to do anything about this unless it bites you. If it does,
    one optional line at the top of your script, after the imports, undoes it:

    ```python
    import signal

    signal.siginterrupt(signal.SIGINT, True)
    ```

!!! note "Platforms"
    `local_pool` is **POSIX only** and refuses to be created elsewhere. The rest
    of `ropt` is not known to be broken on Windows, but it is not tested there.
    Free-threaded (no-GIL) builds of Python are untested and unsupported.


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
evaluates one point at a time, while a larger pool (or a process, local, or HPC
pool) evaluates several at once. Without a pool the runs evaluate on their own
driver threads, so your objective is then called by several threads at once.

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
workers on a process, local, or HPC pool. Without a pool, `offload` simply runs
the function inline instead, so your code never needs a separate fallback for
the case where no pool is available. See
[Parallel Execution and Many Runs](../running/parallel.md#offloading-your-own-work) for
details.
