# Parallel Execution and Many Runs

Evaluations within one optimization can run in parallel, whole optimizations
can run at once, and work of your own can be offloaded the same way. All three
go through a **pool**, opened from a [`session`][ropt.simple.session].

## Running in parallel

By default [`optimize`][ropt.simple.optimize] runs on the calling thread, one
evaluation at a time. To
run the evaluations in parallel, open a [`session`][ropt.simple.session], ask it
for a **pool**, and pass that pool to the run. [Running in
Parallel](../getting_started/execution.md) introduces the five kinds; this page
is the full account of each:

```python
from ropt.simple import session

with session() as s:
    result = optimize(config, x0, objective, pool=s.thread_pool(workers=8))
```

The runnable script for this section is
[examples/simple/parallel.py](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/parallel.py),
which takes `-m` to swap its thread pool for a process pool.

A session is a background event loop that the pools run on. Closing it releases
every pool it created, so that is normally all the cleanup you need. Nothing is
implicit: a run evaluates on the pool you hand it, and on no other. A run given
no pool evaluates in-process, wherever it is called from — including from a
thread you started yourself.

### How many workers?

`ropt` parallelizes two things, and only these two: the **evaluations within one
optimization batch**, and whole **optimizations against each other**. Nothing
else overlaps — a single optimization is a sequence of batches, and the next one
cannot start before the current one is complete.

So the number of workers worth asking for is roughly

```
batch size  ×  optimizations running at once
```

capped by what the machine or the queue will actually give you. Beyond that
figure the extra workers have nothing to do and sit idle.

Batch size follows from the problem, not from a setting: it is how many
evaluations the optimizer asks for at once. For a gradient-based run over an
ensemble that is one per realization, plus their perturbations on the batches
where a gradient is estimated. The second factor is `1` unless you use
[`optimize_many`][ropt.simple.optimize_many], where it is the `limit` argument
(or the number of runs, if you set no limit).

A worker is not free, so this is an upper bound rather than a target. Ask for
what the work needs; the useful number is usually much smaller than the
machine's core count.

!!! tip "How a batch is split across workers"
    Each evaluation in a batch is transferred to a worker as its own task by
    default, which spreads the batch as widely as the pool allows. Every
    transfer costs something, though, so when the evaluations are cheap the
    transfers can dominate. Set `bundle_size=` on the pool to group several
    evaluations into one task, or `bundle_size=0` to send the whole batch as a
    single task. The evaluations within a task run one after another, so `0`
    gives up parallelism inside the batch entirely: it is for a pool whose
    parallelism comes from the runs above it, as in
    [Nested Optimization](nested.md#two-pools-not-one).

    `workers` and `bundle_size` are the two halves of matching work to capacity.
    `workers` says how many tasks may be in flight; `bundle_size` says how much
    work one task should carry. With a batch of 100 cheap evaluations and
    8 workers, the default sends 100 separate tasks and pays 100 transfer costs
    to keep 8 workers busy; `bundle_size=13` sends 8 and pays 8. Raise it when
    the evaluations are cheap relative to a transfer — above all on a
    `process_pool`, a `local_pool` or an `hpc_pool`, where a transfer means
    copying data and starting something. Leave it at `1` when they are
    expensive, or when they vary in cost and bundling would leave one worker
    holding all the slow ones.

You can keep several pools open at once and choose per run:

```python
with session() as s:
    fast = s.thread_pool(workers=8)
    heavy = s.process_pool(workers=4)
    cheap = optimize(config, x0, objective, pool=fast)
    costly = optimize(config, x0, expensive_objective, pool=heavy)
```

!!! note "Pools inside an evaluation"
    An evaluation function may start a run of its own, but open its pool
    **once**, in the calling code, and pass it to every evaluation. Opening one
    per evaluation gives each a budget nobody else knows about: ten evaluations
    each opening a ten-worker pool put a hundred workers on the machine, where
    one shared pool puts ten. Which pool it may be is a separate question, and
    the rules are in
    [Pools inside an evaluation](#pools-inside-an-evaluation) below.

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

    A closed pool cannot be reopened, and a run still using it returns with
    [`ExitCode.EXECUTOR_STOPPED`][ropt.enums.ExitCode] rather than raising —
    though on a thread pool the evaluations already running still finish first,
    since a thread cannot be interrupted; see
    [Stopping a run](#stopping-a-run). Starting a *new* run
    on it is refused before anything runs, with a
    [`WorkflowError`][ropt.exceptions.WorkflowError] saying the pool is closed —
    which is what you get if a pool outlives the `with session()` block that
    created it.

### Evaluating on threads { #thread-pool }

A [`thread_pool`][ropt.simple.Session.thread_pool] runs the evaluations on
background threads inside your own process. Nothing is copied, so any Python
function works as the objective and it can freely use the data around it:

```python
with session() as s:
    result = optimize(config, x0, objective, pool=s.thread_pool(workers=4))
```

Use it when each evaluation spends most of its time **waiting** — starting an
external program, reading a file, calling a network service. While one waits,
the others run.

Threads share one Python interpreter, so arithmetic written in Python itself
does not get faster on more threads. Array libraries are a different matter:
`numpy` and its kin do their work outside Python and let the other threads run
meanwhile. "My objective computes" is therefore not on its own a reason to reach
past this pool — see [Which pool should I use?](#which-pool).

### Evaluating in worker processes { #process-pool }

A [`process_pool`][ropt.simple.Session.process_pool] runs the evaluations in a
handful of separate processes, reused across the run. Each has its own
interpreter, so this is where heavy Python computation actually gets faster:

```python
with session() as s:
    result = optimize(config, x0, objective, pool=s.process_pool(workers=4))
```

Reach for it when the computation is **Python code**, or when each evaluation
needs its own copy of something a library keeps globally. An objective that
mostly runs an external program gains nothing here that a thread pool would not
have given more cheaply.

The objective and its data are **copied** to the workers, so they must be
serializable. An objective defined at module level — or in the script you ran,
which each worker re-imports — works as is; a lambda, a closure, or a
function defined in a notebook cell needs the `cloudpickle` extra (see
[Installation](../getting_started/installation.md#optional-extras)). Results can
only come **back** through the return value; see
[Handlers and the process boundary](handlers.md#handlers-and-the-process-boundary).

!!! warning "This pool does not clean up programs your objective started"
    When a run is stopped — by Ctrl-C, or by closing the pool — the worker
    processes are killed, but anything they had launched themselves is not: a
    simulator or solver started by your objective keeps running, unattached,
    after your program is gone. Nothing warns you. If your objective launches
    external programs, use a `local_pool` instead, which was built for exactly
    this.

### Running each evaluation as a local job { #local-pool }

A [`local_pool`][ropt.simple.Session.local_pool] runs each evaluation as a
separate process on this machine, with its output captured to a file. It needs
no extras and no configuration, and it is the same shape as an `hpc_pool` minus
the scheduler — so an objective that works on one works on the other:

```python
from ropt.simple import session

with session() as s:
    result = optimize(config, x0, objective, pool=s.local_pool(workers=4))
```

| Parameter     | Description                                                                |
| ------------- | ------------------------------------------------------------------------- |
| `workers`     | Maximum number of concurrent local jobs (default: 1).                     |
| `workdir`     | Directory holding each evaluation's files. Defaults to a temporary directory the pool removes again, unless something is left in it to read. |
| `retries`     | Extra polls to wait for a result (default: 0, which is enough).           |
| `bundle_size` | Evaluations bundled into one local process, `0` for the whole batch as one (default: 1). See [How a batch is split across workers](#how-many-workers) above. |

Two things distinguish it from a `process_pool`, and both matter when an
evaluation is a job rather than a function call:

- **Stopping reaches what the evaluation started.** Each job runs in a process
  group of its own, so cancelling one signals the simulator or solver it
  launched as well. A `process_pool` kills only its own workers and orphans the
  rest.
- **Output survives failure.** Whatever the evaluation printed is captured, and
  the last lines are attached to the error, which is often the only trace a job
  that died before returning anything leaves behind.

!!! note "Where the working directory goes"
    With no `workdir`, the pool works in a temporary directory of its own and
    removes it when it closes — but only when there is nothing left in it worth
    reading. If an evaluation **failed**, its captured output is kept, so the
    directory is kept with it and its path is logged:

    ```
    WARNING  ropt.components.executors: Keeping the local working directory
             /tmp/ropt-local-8f3a1c: a work item failed.
    ```

    A `workdir` you pass yourself is never removed, which is the way to choose
    the location rather than be told it:

    ```python
    pool = s.local_pool(workers=4, workdir="/scratch/my-run")
    ```

    Give each pool that runs at the same time a directory of its own; files are
    named after the evaluations, and the pool refuses to overwrite one that
    already exists.

This pool needs process groups and is therefore **POSIX only**: creating it on
another platform raises an [`ExecutionError`][ropt.exceptions.ExecutionError]
rather than quietly giving a weaker guarantee.

#### Worker or job: `process_pool` or `local_pool`? { #process-or-local }

Both run your objective in a separate process, so "is it copied?" does not tell
them apart. What differs is whether a process is a *worker* or a *job*:

|  | `process_pool` | `local_pool` |
| --- | --- | --- |
| Processes | a few, **reused** for many evaluations | one **fresh** process per evaluation |
| Start-up cost | paid once, when the pool opens | paid again on every evaluation |
| Programs your objective starts | keep running when the run stops | killed along with the evaluation |
| Output of a failed evaluation | lost | captured, and attached to the error |
| Sending the objective | your script is re-imported, so a function defined in it can be found by name | a fresh command; needs an importable module or `ropt[cloudpickle]` |
| Platform | anywhere | POSIX only |

One question decides it: **is an evaluation a function call, or a job?** A call
is too short to pay for a process each time, so reuse a few workers and take
`process_pool`. A job runs a simulator, writes files, and lasts long enough that
one process start is negligible — take `local_pool`, or an `hpc_pool` if it belongs
on a cluster.

!!! tip "Heading for a cluster? Develop on a `local_pool`"
    A `local_pool` is an `hpc_pool` without the scheduler: both run each
    evaluation as a job, one fresh process at a time, and both capture its
    output and attach the tail to the error. An objective that runs on one runs
    on the other, so you can get the job itself right on your own machine —
    without a queue, the `ropt[hpc]` extra, or a cluster configuration — and
    switch when it works:

    ```python
    pool = s.local_pool(workers=4)     # develop and test here
    pool = s.hpc_pool(workers=100)     # then run here
    ```

    A `process_pool` is the wrong rehearsal: it reuses workers, keeps your
    objective in Python, and leaves the programs it starts running when the run
    stops — none of which is how a cluster job behaves.

### Running on an HPC cluster

An [`hpc_pool`][ropt.simple.Session.hpc_pool] submits each evaluation as a job to
an HPC queue through [`pysqa`](https://pysqa.readthedocs.io/); it needs the
`ropt[hpc]` extra. With no further arguments it uses the default cluster and
queue from the `pysqa` configuration of your `ropt` installation:

```python
from ropt.simple import session

with session() as s:
    result = optimize(config, x0, objective, pool=s.hpc_pool(workers=10))
```

The runnable script is
[examples/simple/hpc.py](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/hpc.py).
Pass it `--local` and it runs the identical optimization on a `local_pool`,
which is the rehearsal described above, so the example works with or without a
cluster to hand.

A job is nothing more than a submission script with your evaluation command in
it, and there are **two mutually exclusive ways** to say what that script should
be: a `pysqa` configuration, or a `template` you write yourself.

#### Using an installed configuration

This is the usual case. The configuration already describes the clusters and
their queues, so all you do is pick one and say how much of it you want:

```python
pool = s.hpc_pool(workers=10, queue="long", cores=4)
```

`queue` names a queue **defined in the configuration**, which is not necessarily
your scheduler's partition name — it selects a configured entry, and that
entry's script decides which partition the job lands on. Ask your site which
queues exist, or read them off the configuration.

When the configuration defines several clusters, `cluster` picks one:

- Give `cluster` to select it directly; adding `queue` requires that queue to
  exist on it.
- Give only `queue` and the cluster providing it is found automatically, which
  needs exactly one cluster to provide it — no match, or several, is an error.
- Give neither and the configuration's own defaults apply.

`config_path` points at a configuration other than the installed one. See
[HPCExecutor](../advanced/parallel.md#hpcexecutor) for how such a directory is
laid out and where the installed one lives.

#### Asking for resources

`cores`, `memory_max` and `run_time_max` are passed to the submission script,
and `submit_options` carries anything else that script declares:

```python
pool = s.hpc_pool(
    workers=10,
    queue="long",
    cores=4,
    memory_max=16,
    run_time_max=7200,
    submit_options={"account": "my-project"},
)
```

For `account` to have any effect the script must reference it. A variable a
script never mentions is simply ignored, and one the script mentions but nobody
supplies renders as empty — so a misspelling on either side drops the directive
silently rather than failing. Entries that are `None` are dropped, so omitting a
key and passing `None` mean the same thing. A name the executor sets itself,
such as `cores` or `queue`, is rejected rather than allowed to override it.

With a configuration, `cores` and `run_time_max` are also **clamped** to the
selected queue's limits rather than rejected: asking for more cores than the
queue allows quietly gets you the queue's maximum.

#### Submitting with your own template

A `template` is simply the script that gets run on the cluster, written by you
instead of taken from a configuration. Since there is no configuration to say
what kind of cluster this is, `scheduler` tells ropt which queueing system to
submit to — that is what decides whether it runs `sbatch` or `bsub`. It defaults
to `"slurm"`.

Nothing else is resolved for you: **the queue is not an argument here**, it has
to be written into the script, along with everything else the scheduler needs.
For Slurm that looks like this — other systems use entirely different
directives:

```python
TEMPLATE = """\
#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name={{job_name}}
#SBATCH --output={{output}}
#SBATCH --chdir={{working_directory}}
#SBATCH --ntasks={{cores}}
{%- if memory_max %}
#SBATCH --mem={{memory_max}}G
{%- endif %}

{{command}}
"""

pool = s.hpc_pool(workers=10, template=TEMPLATE, scheduler="slurm", cores=4)
```

The script is a [Jinja](https://jinja.palletsprojects.com/en/stable/templates/)
template, rendered by `pysqa` through the `jinja2` package. `{{name}}`
inserts a value and `{% if name %}...{% endif %}` leaves a line out when none was
given, which is how the memory directive above disappears unless `memory_max` is
set. The values available are the arguments described above — `job_name`,
`output`, `working_directory`, `cores`, `memory_max`, `run_time_max`, `command`
— plus whatever you pass in `submit_options`.

Two of them are worth getting right: `{{command}}` is your evaluation and the
script does nothing without it, and `{{output}}` is the file ropt reads back to
explain a failed job. A script that omits `--output={{output}}` still runs, but a
job that dies takes the only explanation with it.

Because a template submits without a configuration, it **cannot be combined**
with `config_path`, `cluster` or `queue`; passing them together raises a
`ValueError` when the pool is created.

`hpc_pool` accepts the following parameters:

| Parameter     | Description                                                                |
| ------------- | ------------------------------------------------------------------------- |
| `workers`     | Maximum number of concurrent cluster jobs (default: 1).                   |
| `cores`       | Number of CPUs per job (default: 1).                                      |
| `cluster`     | Cluster name, when the `pysqa` config defines several.                    |
| `queue`       | Name of a queue defined in the configuration.                             |
| `workdir`     | Shared-filesystem working directory (defaults to the current directory).  |
| `config_path` | The `pysqa` configuration directory.                                      |
| `template`    | A submission-script template, used instead of a configuration.            |
| `scheduler`   | The queueing system a `template` is written for; only meaningful with one. |
| `memory_max`  | Memory per job.                                                           |
| `run_time_max` | Run time per job, typically in seconds.                                  |
| `submit_options` | Extra variables for the submission script. `None` entries are dropped. |
| `retries`     | Extra polls to wait for a result that is missing or unreadable (default: 30). |
| `bundle_size` | Evaluations bundled into one cluster job, `0` for the whole batch as one job (default: 1). See [How a batch is split across workers](#how-many-workers) above. |

### Evaluating in-process, on purpose

[`serial_pool`][ropt.simple.serial_pool] is a pool with no workers: it carries
only the batch-ID counter that the runs sharing it draw from, and their
evaluations happen in-process on the calling thread. It needs no session, and
needs no releasing.

Use it to give several runs one continuous batch-ID sequence without running
their evaluations in parallel, or simply to say in the code that a run is meant
to evaluate in-process.

### Which pool should I use? { #which-pool }

[Running in Parallel](../getting_started/execution.md) asks the question that
rules choices *out*: whether your objective touches anything beyond its
arguments and its return value. If it does, stay with threads or with no pool,
because the others work on a copy. Once that is settled, the choice is about
speed:

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
    work depends — but treat it as a place to start, and decide on the timings.

??? tip "If more workers makes it *slower*"
    `numpy`, `scipy` and similar are already multi-threaded underneath, through
    a BLAS library that by default takes **every core on the machine**. Run four
    evaluations at once and you have four such libraries each doing that: the
    machine is oversubscribed several times over, the threads fight for cores,
    and everything slows down.

    The symptom is misleading, because it reads as "parallelism does
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

### Pools inside an evaluation { #pools-inside-an-evaluation }

An evaluation function may start a run of its own — that is how
[Nested Optimization](nested.md) works — and give it a pool, on two conditions.

It must be a **different** pool. A nested run waits for its own evaluations to
finish, so one handed the pool it is already running on would wait for the
workers it is itself occupying — a deadlock as soon as they are all busy, which
is the normal case, since a run fills its pool with one work item per
realization. Rather than hang, the pool refuses work submitted by the evaluation
itself with a [`WorkflowError`][ropt.exceptions.WorkflowError]. A thread the
evaluation starts is on its own: it is not recognized as a worker, so it can
still deadlock on the pool. Give the inner run its own pool, or a
[`serial_pool`][ropt.simple.serial_pool], which evaluates inline and can always
be reused.

The evaluation must stay **in your process**, so on a thread pool or a serial
pool. On a process, local, or HPC pool the evaluation function is copied into a
worker, and a pool cannot be copied with it: build the inner pool inside the
worker, from a session opened there, or run the inner optimization without one.
An evaluation function that carries a pool along anyway is refused when the work
item is sent, rather than failing somewhere deep inside the run.

## Stopping a run

Press Ctrl-C, or close the pool, and `ropt` stops handing out new work at once.
What happens to the evaluations already running depends on the pool, because
what *can* be done to them differs:

| Pool | Evaluations already running |
| --- | --- |
| none / `serial_pool` | the current one finishes |
| `thread_pool` | they **run to completion** — a thread cannot be interrupted |
| `process_pool` | the worker processes are **killed**, but not what they launched |
| `local_pool` | each evaluation **and everything it launched** is killed |
| `hpc_pool` | the jobs are **deleted from the queue** |

Two consequences follow.

**A thread pool cannot be hurried.** Python provides no way to interrupt a
running thread from outside, so a long evaluation on a `thread_pool` ends when
it ends, and your program cannot exit before it does. `ropt` logs a warning
naming how many evaluations it is waiting for, because the wait is otherwise
indistinguishable from a hang. If an evaluation may run long and
has to be interruptible, put it on one of the other pools.

**Stopping is a request, not a guarantee.** On the pools that kill,
everything is asked to end and not waited for. A program that ignores the
request, or that is stuck inside the operating system, keeps running. The run
exits promptly instead of waiting out the current batch, but processes it
started may still be alive afterwards.

!!! tip "If Ctrl-C seems to do nothing at all"
    Some third-party packages change a process-wide setting when imported that
    stops Ctrl-C from breaking into a program that is *waiting* — and it then
    affects your whole program, not just `ropt`. Importing `ropt.simple` is
    enough to trigger it. One line undoes it; see
    [Keyboard Interrupts](../troubleshooting/keyboard_interrupt.md).

!!! note "Platforms"
    `local_pool` is **POSIX only** and refuses to be created elsewhere. The rest
    of `ropt` is not known to be broken on Windows, but it is not tested there.
    Free-threaded (no-GIL) builds of Python are untested and unsupported.

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
  `process_pool`, `local_pool`, or `hpc_pool` — runs several evaluations at
  once.

Sharing one pool is also what keeps the runs' batch IDs apart, since they draw
from its single counter. It is one budget as well: `workers=10` means ten
evaluations at a time across the whole batch of runs, not ten per run.

[examples/simple/optimize_many.py](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/optimize_many.py)
runs one optimization per start vector, capping how many go at once and tagging
each with its own metadata:

```python
--8<-- "examples/simple/optimize_many.py:run"
```

The two callback arguments differ in the same way. `report=` is **per run**: one
callback watches every run, or pass a list with one callback per run. `handlers=`
is **shared**: one list of groups that all runs feed together, which is why a
plain handler is refused there — see [Sharing a handler across concurrent
runs](handlers.md#sharing-a-handler-across-concurrent-runs).

!!! warning "One `report=` callback is called by every run at once"
    A single callback is wired into each run separately, and each run calls it
    on its own thread. Nothing serializes those calls, so a callback that
    appends to a list, updates a counter, or writes a file needs a lock of its
    own. Give each run its own callback when they must stay apart, or collect
    the results in a [shared
    group](handlers.md#sharing-a-handler-across-concurrent-runs), where the
    dispatcher serializes them for you.

!!! warning "A shared group makes the runs wait for each other"
    That serialization is not free. A group processes events **one at a time**,
    in submission order, and the run that emitted one waits until every handler
    has finished with it. Never seeing two results at once is exactly what makes
    a handler safe to share — but it means a slow handler throttles the whole
    batch, since every run queues behind the others, once per result produced.

    So keep shared handlers cheap. If one must do slow work — writing a file,
    talking to a database — register it with
    [`threaded`](handlers.md#running-a-handler-in-a-thread), which moves it off
    the session's event loop so the pools keep working meanwhile. That does
    **not** make handling concurrent: the events are still processed one at a
    time.

!!! warning "Without a pool the driver threads do the evaluating"
    `optimize_many` needs no session and no pool. Without one, the runs still
    execute concurrently, but each evaluates in-process on its own driver
    thread — so your evaluation function is called by several threads at once
    and must tolerate that. Give the call a pool, or a
    [`serial_pool`][ropt.simple.serial_pool] if you want one shared batch-ID
    sequence, when it must not be.

!!! warning "Not every backend can take part"
    An optimizer that needs a working directory of its own, writes to a file
    whose name is fixed, or keeps state inside its library between calls cannot
    run while anything else is running in the same process — another run of its
    own kind included. Each backend documents whether this applies to it.
    Select it as
    [`external/...`](#external-backend) and
    it gets a process of its own, where none of that is shared.

    Optimizer output capture is likewise for one run at a time. If more than
    one of these runs sets
    [`stdout` or `stderr`](../optimizer_setup/configuration_sections.md#optimizer), the
    second to start raises [`WorkflowError`][ropt.exceptions.WorkflowError].
    Leave both unset here, and set [`verbose=False`](../optimizer_setup/configuration_sections.md#backend)
    unless you want the runs' reports interleaved on the terminal.

### Failure in one run

The first run to raise propagates its exception immediately (fail-fast). Runs
that have not started yet are skipped, but a run already in progress cannot be
stopped from the outside: it is abandoned, and keeps going until it finishes on
its own — so returning after a failure can still take as long as a full
optimization. Closing the pool cuts that short: the abandoned run then stops at
its next evaluation and returns rather than raising, usually with
[`ExitCode.EXECUTOR_STOPPED`][ropt.enums.ExitCode], though a run that ends its
own optimizer loop first reports that reason instead. Either way its result is
discarded.

## Running the optimizer in a separate process { #external-backend }

Prefix the method with `external/` to run the optimization algorithm in a
process of its own:

```python
"backend": {"method": "external/scipy/slsqp"}
```

`ropt` spawns a child process, creates the named backend there, and lets it
drive the optimization. The function and gradient evaluations still happen in
the original process: the child sends each set of variables back, the parent
evaluates it as usual, and the values are passed to the child. An error raised
in the child is re-raised in the parent.

This is useful when a backend cannot safely share a process with the rest of
your program — for example one that crashes the interpreter, leaks memory,
keeps state between runs, or links against native libraries that clash with
your other dependencies.

It is also the answer for a backend that **cannot run concurrently in-process**.
Some optimizers need a working directory of their own, write to a file whose
name is fixed, or keep state inside the library that a second simultaneous run
corrupts. What such a backend rules out is not merely a second run of its own
kind: changing the working directory applies to the whole process, so it breaks
another run's relative output path, and any file your evaluation function opens
by relative name, just as surely. Each backend states in its own documentation
whether this applies to it; where it does, `external/` is what lets it run
alongside anything else, because the state it needs is then its own. This
matters as soon as runs overlap — see [Many optimizations at
once](#many-optimizations-at-once).

Two details differ from the other backends:

- The method must name the delegate in full, as `external/plugin/method` or
  `external/method`. The `external/` prefix is removed and the rest is resolved
  like any other method string. `external` is never selected implicitly, so it
  is used only when you ask for it by name.
- The problem is sent to the child process, so everything describing it must be
  serializable. The built-in plugins are, and so is any plugin class defined in
  a module that can be imported. Only if you pass a plugin instance of a class
  defined inside a function or a notebook do you need the optional
  `cloudpickle` extra (see
  [Installation](../getting_started/installation.md#optional-extras)). Without
  it the two differ in *where* they fail: a class defined inside a function
  cannot be sent at all, and is refused here with an
  [`ExecutionError`][ropt.exceptions.ExecutionError]; a class defined in a
  notebook is sent by name, and the failure arrives from the child, which
  reports the name it could not find. Your objective function is never
  affected: it stays in this process.

This has nothing to do with evaluating in parallel; for that see [Running in
Parallel](../getting_started/execution.md).

## Offloading your own work

You can hand **your own** functions to a pool with
[`offload`][ropt.simple.offload]. It is useful when code you control — a custom
step, a custom component, or a helper you call between optimizations — has an
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

As with the evaluation function on a process, local, or HPC pool, the callables
and their arguments are **copied to the workers**, since they run in separate
processes.

!!! warning "Offloaded work coordinates with nothing"
    An offloaded callable runs wherever its pool puts it, and on a process,
    local, or HPC pool that is somewhere else. It may create handlers and pools
    of its own, but they are *its* handlers and *its* pools.

    - **Results cannot be tracked across offloaded calls.** A handler created
      inside one sees only that call's results and cannot be brought back:
      handlers refuse to cross a process boundary, so the return value is all
      that comes back. Following several concurrent pieces of work in one place
      is something [`optimize_many`](#many-optimizations-at-once) can do and
      `offload` cannot.
    - **Worker budgets multiply, with no way around it.** A pool cannot be
      carried into a worker process, so a callable that needs one opens its own.
      Four callables that each open a ten-worker pool put **forty** workers on
      the machine in four independent groups — and they do not pool their
      effort: one with twenty pieces of work still runs ten at a time while
      another group sits idle. The machine carries forty workers, and never
      forty working on the same thing. Where evaluations stay in your process
      this is avoidable by sharing one pool; offloaded onto a process, local, or
      HPC pool it is not.

    So reach for `offload` when a piece of work is genuinely self-contained and
    hands its answer back as a return value. When several pieces must be
    coordinated — counted, collected, or held to one budget — drive them from
    your own process instead.

### Without a pool

`offload` with no pool — or with a [`serial_pool`][ropt.simple.serial_pool] —
runs the callables inline, on the calling thread. So code that may or may not
have a pool to hand needs no guard and no fallback: pass along whatever it has,
including `None`.

```python
def transform(x, pool=None):
    return offload(partial(expensive, x), pool=pool)
```

!!! note "An inline handler in a shared group cannot offload"
    A handler in a [shared group](handlers.md#sharing-a-handler-across-concurrent-runs)
    that runs inline is on the session's event loop; offloading to a pool on
    that same session would starve the very loop it is waiting on, so it raises
    a [`WorkflowError`][ropt.exceptions.WorkflowError]. A
    [`threaded`](handlers.md#running-a-handler-in-a-thread) handler runs on a dispatcher
    worker instead and can offload, as can a local handler, which runs on the
    thread driving the run.

    Better still, do parallel work from your optimization code and leave
    handlers to handle results.
