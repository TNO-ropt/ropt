# Parallel Execution and Many Runs

This page continues [Running Optimizations](running.md): running evaluations
in parallel, running several optimizations at once, and offloading your own
work to a pool.

## Running in parallel

By default `optimize` runs on the calling thread, one evaluation at a time. To
run the evaluations in parallel, open a [`session`][ropt.simple.session], ask it
for a **pool**, and pass that pool to the run. See
[Running in Parallel](../getting_started/execution.md) for a full explanation of the
choices and their trade-offs:

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
what the work needs; the interesting number is usually a good deal smaller than
the machine's core count.

!!! tip "How a batch is split across workers"
    Each evaluation in a batch is transferred to a worker as its own task by
    default, which spreads the batch as widely as the pool allows. Every
    transfer costs something, though, so when the evaluations are cheap the
    transfers can dominate. Set `bundle_size=` on the pool to group several
    evaluations into one task, or `bundle_size=0` to send the whole batch as a
    single task. The evaluations within a task run one after another, so `0`
    gives up parallelism inside the batch entirely: it is for a pool whose
    parallelism comes from the runs above it, as in the note below.

    `workers` and `bundle_size` are the two halves of matching work to capacity.
    `workers` says how many tasks may be in flight; `bundle_size` says how much
    work one task is worth carrying. With a batch of 100 cheap evaluations and
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
    serial pool. On a process, local, or HPC pool the evaluation function is
    copied into a worker, and a pool cannot be copied with it: build the inner
    pool inside the worker, from a session opened there, or run the inner
    optimization without one. An evaluation function that carries a pool along
    anyway is refused when the work item is sent, rather than failing somewhere
    deep inside the run.

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
    [`ExecutorStopped`][ropt.exceptions.ExecutorStopped] — though on a thread
    pool the evaluations already running still finish first, since a thread
    cannot be interrupted; see
    [Stopping a run](../getting_started/execution.md#stopping-a-run). Starting a *new* run
    on it is refused before anything runs, with a
    [`WorkflowError`][ropt.exceptions.WorkflowError] saying the pool is closed —
    which is what you get if a pool outlives the `with session()` block that
    created it.

### Running each evaluation as a local job

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
[HPCExecutor](../workflows/parallel.md#hpcexecutor) for how such a directory is
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
key and passing `None` mean the same thing.

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
