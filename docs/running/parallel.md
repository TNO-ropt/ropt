# Parallel Execution and Many Runs

Evaluations within one optimization can run in parallel, whole optimizations
can run at once, and work of your own can be offloaded the same way. All three
go through an **executor**.

## Running in parallel

By default [`optimize`][ropt.simple.optimize] runs on the calling thread, one
evaluation at a time. To run the evaluations in parallel, build an **executor**
and pass it to the run. [Running in
Parallel](../getting_started/execution.md) introduces the four kinds; this page
is the full account of each:

```python
from ropt.simple import ThreadExecutor

with ThreadExecutor(workers=8) as executor:
    result = optimize(config, x0, objective, executor=executor)
```

The runnable script for this section is
[examples/simple/parallel.py](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/parallel.py),
which takes `-m` to swap its thread executor for a process executor.

An executor is a context manager: leaving the `with` block releases its
workers. Nothing is implicit: a run evaluates on the executor you hand it, and
on no other. A run given no executor evaluates in-process, wherever it is
called from — including from a thread you started yourself.

### How many workers?

`ropt` parallelizes two things, and only these two: the **evaluations within one
optimization batch**, and whole **optimizations against each other**. Nothing
else overlaps — a single optimization is a sequence of batches, and the next one
cannot start before the current one is complete.

So the number of workers that can be kept busy is roughly

```
batch size  ×  optimizations running at once
```

capped by what the machine or the queue will actually give you. Beyond that
figure the extra workers have nothing to do and sit idle.

Batch size follows from the problem, not from a setting: it is how many
evaluations the optimizer requests at once. For a gradient-based run over an
ensemble that is one per realization, plus their perturbations on the batches
where a gradient is estimated. The second factor is `1` unless you use
[`optimize_many`][ropt.simple.optimize_many], where it is the `limit` argument
(or the number of runs, if you set no limit).

A worker is not free, so this is an upper bound rather than a target. Ask for
what the work needs; the useful number is usually much smaller than the
machine's core count.

!!! tip "How a batch is split across workers"
    Each evaluation in a batch is transferred to a worker on its own by
    default, which spreads the batch as widely as the executor allows. Every
    transfer costs something, though, so when the evaluations are cheap the
    transfers can dominate. Pass `bundle_size=` to
    [`optimize`][ropt.simple.optimize], [`optimize_many`][ropt.simple.optimize_many],
    [`evaluate`][ropt.simple.evaluate] or
    [`evaluate_many`][ropt.simple.evaluate_many] to send several evaluations to
    a worker together, or `bundle_size=0` to send a whole batch at once. The
    evaluations in one bundle run after each other, so `0` gives up parallelism
    inside the batch entirely: it is for a run whose parallelism comes from the
    layer above it, as in
    [Nested Optimization](nested.md#two-executors-not-one).

    `workers` and `bundle_size` are the two halves of matching work to capacity,
    one on the executor and one on the run. `workers` sets how many bundles may
    be in flight; `bundle_size` sets how much work one of them carries. With a
    batch of 100 cheap evaluations and 8 workers, the default sends 100 separate
    transfers to keep 8 workers busy; `bundle_size=13` sends 8. Raise it when
    the evaluations are cheap relative to a transfer, which on a
    `ProcessExecutor`, `LocalJobExecutor` or `HPCExecutor` means copying data
    and starting something. Leave it at `1` when the evaluations are expensive,
    or when they vary in cost and bundling would leave one worker holding all
    the slow ones.

    It belongs to the run rather than to the executor because it describes the
    evaluation function, and one executor may serve several. Runs of
    `optimize_many` that differ in cost can each state their own, either as one
    size for every run or as a sequence with one per run:

    ```python
    optimize_many(config, x0, [cheap, costly], executor=executor, bundle_size=[25, 1])
    ```

    Every executor honours it, including a `ThreadExecutor`: a bundle is one
    worker task, so `bundle_size=0` on a thread executor runs the whole batch on
    a single thread. A run without an executor evaluates inline, where a bundle
    has nothing to save.

    An executor also takes a `bundle_size` of its own, used by any run that does
    not state one.

You can keep several executors open at once and choose per run:

```python
with ThreadExecutor(workers=8) as fast, ProcessExecutor(workers=4) as heavy:
    cheap = optimize(config, x0, objective, executor=fast)
    costly = optimize(config, x0, expensive_objective, executor=heavy)
```

!!! note "Executors inside an evaluation"
    An evaluation function may start a run of its own, but build its executor
    **once**, in the calling code, and pass it to every evaluation. Building one
    per evaluation gives each a budget separate from every other: ten evaluations
    each building a ten-worker executor put a hundred workers on the machine,
    where one shared executor puts ten. Which executor it may be is a separate
    question, and the rules are in
    [Executors inside an evaluation](#executors-inside-an-evaluation) below.

!!! tip "Releasing an executor early"
    An executor holds its workers until it is closed. Build it in a `with`
    block, or call `close()`, as soon as you are done with it — above all a
    `ProcessExecutor`, which holds worker interpreters:

    ```python
    for case in cases:
        with ProcessExecutor(workers=4) as executor:
            optimize(config, case, objective, executor=executor)
    ```

    A closed executor cannot be reopened. A run that is *waiting* on it when it
    closes returns with
    [`ExitCode.EXECUTOR_STOPPED`][ropt.enums.ExitCode] rather than raising —
    though on a thread executor the evaluations already running still finish
    first, since a thread cannot be interrupted; see
    [Stopping a run](#stopping-a-run). A run that asks a closed executor for its
    *next* batch is refused with a
    [`WorkflowError`][ropt.exceptions.WorkflowError] saying the executor is
    closed.

### Evaluating on threads { #thread-executor }

A [`ThreadExecutor`][ropt.simple.ThreadExecutor] runs the evaluations on
background threads inside your own process. Nothing is copied, so any Python
function works as the objective and it can freely use the data around it:

```python
with ThreadExecutor(workers=4) as executor:
    result = optimize(config, x0, objective, executor=executor)
```

Use it when each evaluation spends most of its time **waiting** — starting an
external program, reading a file, calling a network service. While one waits,
the others run.

Threads share one Python interpreter, so arithmetic written in Python itself
does not get faster on more threads. Array libraries are a different matter:
`numpy` and its kin do their work outside Python and let the other threads run
meanwhile. "My objective computes" is therefore not on its own a reason to reach
past this executor — see [Which executor should I use?](#which-executor).

### Evaluating in worker processes { #process-executor }

A [`ProcessExecutor`][ropt.simple.ProcessExecutor] runs the evaluations in a
handful of separate processes, reused across the run. Each has its own
interpreter, so this is where heavy Python computation actually gets faster:

```python
with ProcessExecutor(workers=4) as executor:
    result = optimize(config, x0, objective, executor=executor)
```

This executor applies when the computation is **Python code**, or when each
evaluation needs its own copy of something a library keeps globally. An
objective that mostly runs an external program gains nothing here that a thread
executor would not have given more cheaply.

The objective and its data are **copied** to the workers, so they must be
serializable. An objective defined at module level — or in the script you ran,
which each worker re-imports — works as is; a lambda, a closure, or a
function defined in a notebook cell needs the `cloudpickle` extra (see
[Installation](../getting_started/installation.md#optional-extras)). Results can
only come **back** through the return value; see
[Handlers and the process boundary](handlers.md#handlers-and-the-process-boundary).

!!! warning "This executor does not clean up programs your objective started"
    When a run is stopped — by Ctrl-C, or by closing the executor — the worker
    processes are killed, but anything they had launched themselves is not: a
    simulator or solver started by your objective keeps running, unattached,
    after your program is gone. Nothing reports this. If your objective launches
    external programs, use a `LocalJobExecutor` instead, which was built for
    exactly this.

### Running each evaluation as a local job { #local-executor }

A [`LocalJobExecutor`][ropt.simple.LocalJobExecutor] runs each evaluation as a
separate process on this machine, with its output captured to a file. It needs
no extras and no configuration, and it is the same shape as an `HPCExecutor`
minus the scheduler — so an objective that works on one works on the other:

```python
from ropt.simple import LocalJobExecutor

with LocalJobExecutor(workers=4) as executor:
    result = optimize(config, x0, objective, executor=executor)
```

| Parameter     | Description                                                                |
| ------------- | ------------------------------------------------------------------------- |
| `workers`     | Maximum number of concurrent local jobs (default: 1).                     |
| `workdir`     | Directory holding each evaluation's files. Defaults to a temporary directory the executor removes again, unless something is left in it to read. |
| `retries`     | Extra polls to wait for a result (default: 0; a local job writes its result before it exits). |

Two things distinguish it from a `ProcessExecutor`, and both matter when an
evaluation is a job rather than a function call:

- **Stopping reaches what the evaluation started.** Each job runs in a process
  group of its own, so cancelling one signals the simulator or solver it
  launched as well. A `ProcessExecutor` kills only its own workers and orphans
  the rest.
- **Output survives failure.** Whatever the evaluation printed is captured, and
  the last lines are attached to the error, which is often the only trace a job
  that died before returning anything leaves behind.

!!! note "Where the working directory goes"
    With no `workdir`, the executor works in a temporary directory of its own
    and removes it when it closes, unless something in it is still readable. If
    an evaluation **failed**, its captured output is kept, so the
    directory is kept with it and its path is logged:

    ```
    WARNING  ropt.components.executors: Keeping the local working directory
             /tmp/ropt-local-8f3a1c: a work item failed.
    ```

    A `workdir` you pass yourself is never removed, which is the way to choose
    the location rather than be told it:

    ```python
    executor = LocalJobExecutor(workers=4, workdir="/scratch/my-run")
    ```

    Give each executor that runs at the same time a directory of its own; files
    are named after the evaluations, and the executor refuses to overwrite one
    that already exists.

This executor needs process groups and is therefore **POSIX only**: creating it
on another platform raises an
[`ExecutionError`][ropt.exceptions.ExecutionError] rather than quietly giving a
weaker guarantee.

#### Worker or job: `ProcessExecutor` or `LocalJobExecutor`? { #process-or-local }

Both run your objective in a separate process, so "is it copied?" does not tell
them apart. What differs is whether a process is a *worker* or a *job*:

|  | `ProcessExecutor` | `LocalJobExecutor` |
| --- | --- | --- |
| Processes | a few, **reused** for many evaluations | one **fresh** process per evaluation |
| Start-up cost | paid once, when the executor is built | paid again on every evaluation |
| Programs your objective starts | keep running when the run stops | killed along with the evaluation |
| Output of a failed evaluation | lost | captured, and attached to the error |
| Sending the objective | your script is re-imported, so a function defined in it can be found by name | a fresh command; needs an importable module or `ropt[cloudpickle]` |
| Platform | anywhere | POSIX only |

One question separates them: **is an evaluation a function call, or a job?** A call
is too short to pay for a process each time, so reuse a few workers and take
`ProcessExecutor`. A job runs a simulator, writes files, and lasts long enough
that one process start is negligible — take `LocalJobExecutor`, or an
`HPCExecutor` if it belongs on a cluster.

!!! tip "Heading for a cluster? Develop on a `LocalJobExecutor`"
    A `LocalJobExecutor` is an `HPCExecutor` without the scheduler: both run
    each evaluation as a job, one fresh process at a time, and both capture its
    output and attach the tail to the error. An objective that runs on one runs
    on the other, so you can get the job itself right on your own machine —
    without a queue, the `ropt[hpc]` extra, or a cluster configuration — and
    switch when it works:

    ```python
    # Develop and test here:
    executor = LocalJobExecutor(workers=4)
    # Then run here:
    executor = HPCExecutor(workers=100, workdir="/scratch/my-run")
    ```

    A `ProcessExecutor` is the wrong rehearsal: it reuses workers, keeps your
    objective in Python, and leaves the programs it starts running when the run
    stops — none of which is how a cluster job behaves.

### Running on an HPC cluster

An [`HPCExecutor`][ropt.simple.HPCExecutor] submits each evaluation as a job to
an HPC queue through [`pysqa`](https://pysqa.readthedocs.io/); it needs the
`ropt[hpc]` extra. `workdir` is required and must be an existing absolute
directory on a filesystem the compute nodes share. With no further arguments it
uses the default cluster and queue from the `pysqa` configuration of your `ropt`
installation:

```python
from ropt.simple import HPCExecutor

with HPCExecutor(workers=10, workdir="/scratch/my-run") as executor:
    result = optimize(config, x0, objective, executor=executor)
```

The runnable script is
[examples/simple/hpc.py](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/hpc.py).
Pass it `--local` and it runs the identical optimization on a
`LocalJobExecutor`, which is the rehearsal described above, so the example works
with or without a cluster to hand.

A job is nothing more than a submission script with your evaluation command in
it, and there are **two mutually exclusive ways** to say what that script should
be: a `pysqa` configuration, or a `template` you write yourself.

#### Using an installed configuration

This is the usual case. The configuration already describes the clusters and
their queues, so all you do is pick one and say how much of it you want:

```python
executor = HPCExecutor(workers=10, workdir="/scratch/my-run", queue="long", cores=4)
```

`queue` names a queue **defined in the configuration**, which is not necessarily
your scheduler's partition name — it selects a configured entry, and that
entry's script determines which partition the job lands on. Ask your site which
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
executor = HPCExecutor(
    workers=10,
    workdir="/scratch/my-run",
    queue="long",
    cores=4,
    memory_max=16,
    run_time_max=7200,
    submit_options={"account": "my-project"},
)
```

For `account` to have any effect the script must reference it. A variable a
script never mentions is ignored, and one the script mentions but nobody
supplies renders as empty — so a misspelling on either side drops the directive
silently rather than failing. Entries that are `None` are dropped, so omitting a
key and passing `None` mean the same thing. A name the executor sets itself,
such as `cores` or `queue`, is rejected rather than allowed to override it.

With a configuration, resource requests are **clamped** to the selected queue's
limits rather than rejected: asking for more cores than the queue allows quietly
gets you the queue's maximum. `cores` is held to the queue's minimum and
maximum, `run_time_max` to its maximum — and takes that maximum when you give no
value at all — and `memory_max` to its maximum, but only when you pass a number.
A memory string such as `"4G"` is passed to the submission script unchanged.

#### Submitting with your own template

A `template` is the script that gets run on the cluster, written by you
instead of taken from a configuration. Since there is no configuration to say
what kind of cluster this is, `scheduler` names the queueing system to
submit to — that is what determines whether it runs `sbatch` or `bsub`. It defaults
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

executor = HPCExecutor(
    workers=10,
    workdir="/scratch/my-run",
    template=TEMPLATE,
    scheduler="slurm",
    cores=4,
)
```

The script is a [Jinja](https://jinja.palletsprojects.com/en/stable/templates/)
template, rendered by `pysqa` through the `jinja2` package. `{{name}}`
inserts a value and `{% if name %}...{% endif %}` leaves a line out when none was
given, which is how the memory directive above disappears unless `memory_max` is
set. The values available are the arguments described above — `job_name`,
`output`, `working_directory`, `cores`, `memory_max`, `run_time_max`, `command`
— plus whatever you pass in `submit_options`.

Two of them carry the run: `{{command}}` is your evaluation and the
script does nothing without it, and `{{output}}` is the file ropt reads back to
explain a failed job. A script that omits `--output={{output}}` still runs, but a
job that dies takes the only explanation with it.

Because a template submits without a configuration, it **cannot be combined**
with `config_path`, `cluster` or `queue`; passing them together raises a
`ValueError` when the executor is created.

`HPCExecutor` accepts the following parameters:

| Parameter     | Description                                                                |
| ------------- | ------------------------------------------------------------------------- |
| `workers`     | Maximum number of concurrent cluster jobs (default: 1).                   |
| `cores`       | Number of CPUs per job (default: 1).                                      |
| `cluster`     | Cluster name, when the `pysqa` config defines several.                    |
| `queue`       | Name of a queue defined in the configuration.                             |
| `workdir`     | Shared-filesystem working directory; required, and must exist. |
| `config_path` | The `pysqa` configuration directory.                                      |
| `template`    | A submission-script template, used instead of a configuration.            |
| `scheduler`   | The queueing system a `template` is written for; only meaningful with one. |
| `memory_max`  | Memory per job.                                                           |
| `run_time_max` | Run time per job, typically in seconds.                                  |
| `submit_options` | Extra variables for the submission script. `None` entries are dropped. |
| `retries`     | Extra polls to wait for a result that is missing or unreadable (default: 30). |

### Which executor should I use? { #which-executor }

[Running in Parallel](../getting_started/execution.md) asks the question that
rules choices *out*: whether your objective touches anything beyond its
arguments and its return value. If it does, stay with threads or with no
executor, because the others work on a copy. Once that is settled, the choice is
about speed:

| Executor | Where evaluations run | Data | Speeds up heavy Python? | Use when |
| --- | --- | --- | --- | --- |
| none | the calling thread, one at a time | shared | no | evaluations are fast |
| `ThreadExecutor` | background threads, one process | shared | no — one interpreter | each evaluation mostly **waits** (external tool, I/O), or spends its time in `numpy` |
| `ProcessExecutor` | a few reused processes | copied | yes | each evaluation is heavy **Python computation** |
| `LocalJobExecutor` | one process per evaluation | copied | yes | each evaluation is a self-contained **job** on this machine |
| `HPCExecutor` | jobs on a cluster | copied | yes | each evaluation is a big **cluster job** |

??? tip "How to decide, without guessing"
    There is no reliable rule for whether threads will scale on a given
    objective. "I use `numpy`" says almost nothing: whether the GIL is released
    depends on the operation, the dtype and the array size, and a real objective
    is a mixture whose Python-level share is invisible to the person who wrote
    it.

    What makes the answer cheap is an asymmetry: **threads are the cheap thing
    to try, processes are the expensive commitment.** Trying a thread executor
    costs one argument, and its failure mode is *no speedup* — not breakage. So:

    1. Start with `ThreadExecutor`. Time `workers=1` against `workers=4` on a
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

    Then let the executor provide the parallelism instead.

### Executors inside an evaluation { #executors-inside-an-evaluation }

An evaluation function may start a run of its own — that is how
[Nested Optimization](nested.md) works — and give it an executor, on two
conditions.

It must be a **different** executor. A nested run waits for its own evaluations
to finish, so one handed the executor it is already running on would wait for
the workers it is itself occupying — a deadlock as soon as they are all busy,
which is the normal case, since a run fills its executor with one work item per
realization. Rather than hang, the executor refuses work submitted by the
evaluation itself with a [`WorkflowError`][ropt.exceptions.WorkflowError]. A
thread the evaluation starts is on its own: it is not recognized as a worker, so
it can still deadlock on the executor. Give the inner run its own executor, or
none at all, which evaluates inline.

The evaluation must stay **in your process**, so on a thread executor or with no
executor. On a process, local, or HPC executor the evaluation function is copied
into a worker, and an executor cannot be copied with it: build the inner
executor inside the worker, or run the inner optimization without one. An
evaluation function that carries an executor along anyway is refused when the
work item is sent, rather than failing somewhere deep inside the run.

## Stopping a run

Press Ctrl-C, or close the executor, and `ropt` stops dispatching new work at
once. What happens to the evaluations already running depends on the executor,
because what *can* be done to them differs:

| Executor | Evaluations already running |
| --- | --- |
| none | the current one finishes |
| `ThreadExecutor` | they **run to completion** — a thread cannot be interrupted |
| `ProcessExecutor` | the worker processes are **killed**, but not what they launched |
| `LocalJobExecutor` | each evaluation **and everything it launched** is killed |
| `HPCExecutor` | the jobs are **deleted from the queue** |

Two consequences follow.

**A thread executor cannot be hurried.** Python provides no way to interrupt a
running thread from outside, so a long evaluation on a `ThreadExecutor` ends
when it ends, and your program cannot exit before it does. `ropt` warns naming
how many evaluations it is waiting for, because the wait is otherwise
indistinguishable from a hang. If an evaluation may run long and has to be
interruptible, put it on one of the other executors.

**Stopping is a request, not a guarantee.** On the executors that kill,
everything is signalled to end and not waited for. A program that ignores the
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
    `LocalJobExecutor` is **POSIX only** and refuses to be created elsewhere.
    The rest of `ropt` is not known to be broken on Windows, but it is not
    tested there. Free-threaded (no-GIL) builds of Python are untested and
    unsupported.

## Many optimizations at once

To run several optimizations together, use
[`optimize_many`][ropt.simple.optimize_many]. Any of `config`, `x0`, or
`objective` may be a single value (used for every run) or a list (one per run):

```python
from ropt.simple import ThreadExecutor, optimize_many

with ThreadExecutor(workers=4) as executor:
    # One run per start point.
    results = optimize_many(config, start_points, objective, executor=executor)
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
  This is built into `optimize_many` and does not depend on the executor;
  the `limit` argument caps how many run at the same time.
- **The function evaluations** inside those runs all happen on the one executor
  you pass, and the executor determines how they are parallelized. With
  `ThreadExecutor(workers=1)` the runs still progress together, but their
  evaluations are executed one at a time. A larger executor —
  `ThreadExecutor(workers=n)`, `ProcessExecutor`, `LocalJobExecutor`, or
  `HPCExecutor` — runs several evaluations at once.

One executor is one budget: `workers=10` means ten evaluations at a time across
the whole batch of runs, not ten per run. Batch IDs stay distinct whatever you
pass, since every run in the program draws them from one counter.

[examples/simple/optimize_many.py](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/optimize_many.py)
runs one optimization per start vector, capping how many go at once and tagging
each with its own metadata:

```python
--8<-- "examples/simple/optimize_many.py:run"
```

The two callback arguments differ in the same way. `report=` is **per run**: one
callback receives the results of every run, or pass a list with one callback per
run. `handlers=` is **shared**: one list of handlers that all runs feed
together — see [Sharing a handler across concurrent
runs](handlers.md#sharing-a-handler-across-concurrent-runs).

!!! warning "One `report=` callback is called by every run at once"
    A single callback is wired into each run separately, and each run calls it
    on its own thread. Nothing serializes those calls, so a callback that
    appends to a list, updates a counter, or writes a file needs a lock of its
    own. Give each run its own callback when they must stay apart, or pass a
    [handler](handlers.md#sharing-a-handler-across-concurrent-runs) in
    `handlers=`, which takes a lock around every call for you.

!!! warning "A shared handler makes the runs wait for each other"
    That lock is not free. A run that emits a result waits until the handler
    has finished with it, and a second run waits for the first. A slow handler
    therefore throttles the whole batch, once per result produced.

    So keep a shared handler cheap. A handler that must do slow work — writing
    a file, talking to a database — holds up every run feeding it.

!!! warning "Without an executor the driver threads do the evaluating"
    `optimize_many` needs no executor. Without one, the runs still execute
    concurrently, but each evaluates in-process on its own driver thread — so
    your evaluation function is called by several threads at once and must
    tolerate that. Give the call an executor when it must not be.

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
optimization. Closing the executor cuts that short: an abandoned run waiting on
it returns with [`ExitCode.EXECUTOR_STOPPED`][ropt.enums.ExitCode], and one that
asks for its next batch afterwards raises a
[`WorkflowError`][ropt.exceptions.WorkflowError] on its own driver thread.
Either way its result is discarded.

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

You can hand **your own** functions to an executor with
[`offload`][ropt.simple.offload]. It is useful when code you control — a custom
step, a custom component, or a helper you call between optimizations — has an
expensive, self-contained piece of work you want to run on an executor instead
of inline.

Pass a single callable to run one call and get its result back:

```python
from functools import partial

from ropt.simple import ProcessExecutor, offload

with ProcessExecutor(workers=4) as executor:
    result = offload(partial(expensive, data), executor=executor)
```

`offload` takes **zero-argument** callables — bind arguments with
`functools.partial` (or a closure). Pass a **sequence** of callables to run them
concurrently and get a tuple of results in order; they may be entirely different
functions:

```python
with ProcessExecutor(workers=4) as executor:
    first, second = offload(
        [partial(expensive, x), partial(other, y)], executor=executor
    )
```

As with the evaluation function on a process, local, or HPC executor, the
callables and their arguments are **copied to the workers**, since they run in
separate processes.

!!! warning "Offloaded work coordinates with nothing"
    An offloaded callable runs wherever its executor puts it, and on a process,
    local, or HPC executor that is somewhere else. It may create handlers and
    executors of its own, but they are *its* handlers and *its* executors.

    - **Results cannot be tracked across offloaded calls.** A handler created
      inside one sees only that call's results and cannot be brought back:
      handlers refuse to cross a process boundary, so the return value is all
      that comes back. Following several concurrent pieces of work in one place
      is something [`optimize_many`](#many-optimizations-at-once) can do and
      `offload` cannot.
    - **Worker budgets multiply, with no way around it.** An executor cannot be
      carried into a worker process, so a callable that needs one builds its
      own. Four callables that each build a ten-worker executor put **forty**
      workers on the machine in four independent groups — and they do not share
      their effort: one with twenty pieces of work still runs ten at a time
      while another group sits idle. The machine carries forty workers, and
      never forty working on the same thing. Where evaluations stay in your
      process this is avoidable by sharing one executor; offloaded onto a
      process, local, or HPC executor it is not.

    So `offload` fits a piece of work that is genuinely self-contained and
    returns its answer as a return value. When several pieces must be
    coordinated — counted, collected, or held to one budget — drive them from
    your own process instead.

### Without an executor

`offload` with no executor runs the callables inline, on the calling thread. So
code that may or may not have an executor to hand needs no guard and no
fallback: pass along whatever it has, including `None`.

```python
def transform(x, executor=None):
    return offload(partial(expensive, x), executor=executor)
```

!!! warning "Offloading from a handler holds up the runs feeding it"
    A handler runs on the thread driving the run, so it can offload to an
    executor. While it waits, it holds its own lock, so every other run waiting
    on that handler waits too.

    An offloaded callable that reaches back into the same handler — by starting
    a run that carries it in `handlers=` — fails differently per executor. On a
    thread executor it blocks: the handler waits for the offloaded call, and the
    call waits for the lock the handler holds. Without an executor the call runs
    on the handler's own thread and raises a
    [`WorkflowError`][ropt.exceptions.WorkflowError]. On a process executor it
    raises an [`ExecutionError`][ropt.exceptions.ExecutionError], since a
    handler holds a lock and cannot be serialized. See
    [Two hazards](../advanced/workflows.md#two-hazards).

    Work offloaded from the evaluation function runs without that lock held,
    so it does not make the other runs wait.
