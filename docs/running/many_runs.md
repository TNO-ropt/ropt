# Many Runs at Once

Several optimizations can run together rather than one after another, each on
its own driver thread, sharing one pool for their evaluations. Use
[`optimize_many`][ropt.simple.optimize_many]. Any of `config`, `x0`, or
`objective` may be a single value (used for every run) or a list (one per run):

```python
from ropt.simple import session

with session() as s:
    # One run per start point.
    results = s.thread_pool(workers=4).optimize_many(
        config, start_points, objective
    )
```

!!! tip "Give each run an ID"
    Pass a per-run `metadata` list to tag every run with a user-defined
    identifier that travels with its results (and shows up in a
    [`DataFrameHandler`](../results/handlers.md#dataframehandler)'s tables):

    ```python
    labels = ["low", "mid", "high"]
    results = optimize_many(
        config, start_points, objective, metadata=[{"run_id": x} for x in labels]
    )
    for result in results:
        print(result.results.metadata["run_id"])
    ```

    See [Attaching metadata](running.md#attaching-metadata) for details.

## Two levels of concurrency

There are two independent levels of concurrency here:

- **The optimizations** always run concurrently, each on its own driver thread.
  This is built into `optimize_many` and does not depend on the pool;
  the `limit` argument caps how many run at the same time.
- **The function evaluations** inside those runs all happen on the one pool
  the call was started on, and the pool determines how they are parallelized. With
  `thread_pool(workers=1)` the runs still progress together, but their
  evaluations are executed one at a time. A larger pool —
  `thread_pool(workers=n)`, or a process, local or HPC pool — runs several
  evaluations at once.

One pool is one budget: `workers=10` means ten evaluations at a time across
the whole batch of runs, not ten per run. Batch IDs stay distinct whatever you
pass, since every run in the program draws them from one counter.

[examples/simple/optimize_many.py](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/optimize_many.py)
runs one optimization per start vector, capping how many go at once and tagging
each with its own metadata:

```python
--8<-- "examples/simple/optimize_many.py:run"
```

## Watching runs that overlap

The two callback arguments differ. `report=` is **per run**: one
callback receives the results of every run, or pass a list with one callback per
run. `handlers=` is **shared**: one list of handlers that all runs feed
together — see [Sharing a handler across concurrent
runs](../results/handlers.md#sharing-a-handler-across-concurrent-runs).

!!! warning "One `report=` callback is called by every run at once"
    A single callback is wired into each run separately, and each run calls it
    on its own thread. Nothing serializes those calls, so a callback that
    appends to a list, updates a counter, or writes a file needs a lock of its
    own. Give each run its own callback when they must stay apart, or pass a
    [handler](../results/handlers.md#sharing-a-handler-across-concurrent-runs) in
    `handlers=`, which takes a lock around every call for you.

!!! warning "A shared handler makes the runs wait for each other"
    That lock is not free. A run that emits a result waits until the handler
    has finished with it, and a second run waits for the first. A slow handler
    therefore throttles the whole batch, once per result produced.

    So keep a shared handler cheap. A handler that must do slow work — writing
    a file, talking to a database — holds up every run feeding it.

!!! warning "Without a pool the driver threads do the evaluating"
    `optimize_many` needs no pool. Without one, the runs still execute
    concurrently, but each evaluates in-process on its own driver thread — so
    your evaluation function is called by several threads at once and must
    tolerate that. Give the call a pool when it must not be.

!!! warning "Not every backend can take part"
    An optimizer that needs a working directory of its own, writes to a file
    whose name is fixed, or keeps state inside its library between calls cannot
    run while anything else is running in the same process — another run of its
    own kind included. Each backend documents whether this applies to it.
    Select it as
    [`external/...`](../optimizer_setup/optimizer.md#external-backend) and
    it gets a process of its own, where none of that is shared.

    Optimizer output capture is likewise for one run at a time. If more than
    one of these runs sets
    [`stdout` or `stderr`](../optimizer_setup/configuration_sections.md#optimizer), the
    second to start raises [`WorkflowError`][ropt.exceptions.WorkflowError].
    Leave both unset here, and set [`verbose=False`](../optimizer_setup/configuration_sections.md#backend)
    unless you want the runs' reports interleaved on the terminal.

## Failure in one run

A run that raises aborts the other runs on its session. Each of those ends at
its next evaluation boundary with `ABORTED_ON_ERROR`, keeping the best result it
had reached. A run still queued behind `limit` is cut off before its first
evaluation, and reports `ABORTED_ON_ERROR` with no result at all. Nothing is
built for such a run, so an invalid configuration in one is never reported. This
is the default because most runs are started from a script with nobody watching:
a problem ends the script rather than the remaining runs continuing towards
output that will not be used.

A call that fails before it creates any run stops them too. `optimize_many`
broadcasts its arguments first, so lists whose lengths disagree raise
`ValueError`; `evaluate` and `evaluate_batch` raise on a vector of the wrong
shape. Each aborts the other runs on the session before raising.
`RunsFailedError` is not raised in those cases, since no run exists to carry an
outcome. A closed session is not a failure: the call is refused and nothing is
aborted.

The call then raises
[`RunsFailedError`][ropt.exceptions.RunsFailedError]. With several runs there is
no single exception to re-raise and no single set of results to return, so the
error carries both. `outcomes` has one entry per run, in the order the runs were
given, holding either that run's
[`OptimizationResult`][ropt.simple.OptimizationResult] or the exception it
raised:

```python
try:
    results = pool.optimize_many(config, start_points, objective)
except RunsFailedError as failure:
    for index, outcome in enumerate(failure.outcomes):
        if isinstance(outcome, Exception):
            print(f"run {index} raised: {outcome}")
        else:
            print(f"run {index} ended with {outcome.exit_reason.name}")
```

Without this the work the other runs did would be thrown away along with the run
that failed, which matters more now that they are cut off deliberately. The
first exception is chained, so a traceback still shows what went wrong, and a
`KeyboardInterrupt` or `SystemExit` travels on untouched rather than into the
carrier — that is the program going down, not a run reporting a problem. The
interrupt also aborts the runs that are still going and waits for them, so they
end at their next evaluation boundary instead of carrying on unwatched; a second
interrupt abandons them.

The reach is the session, not the call, so a failure also aborts runs that were
started separately on the same session. A run started with the module-level
[`optimize_many`][ropt.simple.optimize_many] has a session of its own, holding
only the runs of that call.

Pass `keep_going=True` to let a run finish anyway, which also lets the runs
queued behind `limit` start:

```python
results = pool.optimize_many(config, start_points, objective, keep_going=True)
```

or `session(keep_going=True)` to make that the default for everything on the
session, which a single run can still override with `keep_going=False`.

The flag decides only whether a run is *aborted*. A run that keeps going still
aborts the others if it fails itself, and its exception still reaches its
caller, so opting out cannot turn a failure into silence.
[`Session.abort`](running.md#stopping-from-outside) reaches every run whatever
the flag says, and those end with `ABORTED` instead: the exit reason
distinguishes an abort that was asked for from one another run caused. See
[Exit Reasons](../results/exit_reasons.md) for what each one means.
