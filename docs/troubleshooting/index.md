# Troubleshooting

A `ropt` optimization keeps going through a failed realization or an infeasible
result, so it can end without a result to return and without raising an error.
Check [`exit_code`](../results/exit_codes.md) before using what comes back. This
page collects the behaviours that most often cause confusion.

Skim it once to know what is here, then come back with a symptom and read the
table of the section it belongs to.

## Results and exit codes

**The returned result is the best *feasible* one.**
[`optimize`][ropt.optimize] returns the best result that satisfies every
constraint to within `constraint_tolerance`, which defaults to `1e-10` and
applies to bounds and linear constraints as well as nonlinear ones. If no
evaluation satisfies them, the optimization ends normally with `result.results`
set to `None`. The evaluations themselves are not lost: every result, feasible
or not, still reaches the [handlers](../results/handlers.md) attached to the
optimization.

**The return value is a summary, not the record of the optimization.** It holds a
single result, the best feasible evaluation. The whole history is available instead:
attach a [`HistoryHandler`][ropt.HistoryHandler] or a
[`DataFrameHandler`][ropt.DataFrameHandler] to collect every result as it
arrives. That history is also what remains when there is no best result to
return.

**`NaN` means "this realization failed", and one failure is already too many.**
[`realization_min_success`](../optimizer_setup/configuration_sections.md#realizations)
defaults to *all* realizations, so a single `NaN` ends the optimization with
`TOO_FEW_REALIZATIONS`. If some realizations are allowed to fail, set it lower:

```python
config = {
    "variables": {"variable_count": 3, "perturbation_magnitudes": 1e-6},
    "realizations": {"weights": [1.0] * 10, "realization_min_success": 8},
}
```

[examples/failures.py](https://github.com/TNO-ropt/ropt/blob/main/examples/failures.py)
runs the same problem twice, once with the default and once allowing the
failure, and prints the exit code and result of each.

**Not everything that goes wrong raises.** Check `exit_code` before using a
result. The exit code gives why the optimization ended; the result is the best
feasible evaluation recorded before it did, and is `None` only when there was
none. An optimization that stops part-way therefore still returns what it had
reached. See [When something goes wrong](../running/running.md#when-something-goes-wrong).

**`FINISHED` does not mean a limit was reached.** It is the exit code
for every optimization the backend ended by itself — because it converged, or
because it reached `max_iterations` or `convergence_tolerance`. Those two live in
the `backend` section and are enforced by the algorithm rather than counted by
`ropt`, so an optimization can stop far short of `max_functions`, and the exit
code does not distinguish which of the three applied. Only `MAX_FUNCTIONS_REACHED` and
`MAX_BATCHES_REACHED` correspond to a limit `ropt` enforced itself.

**A broken worker pool is an error, not a failed realization.** When the pool
itself fails — a worker process is killed, a cluster job never writes its result
— the optimization stops with an [`ExecutionError`][ropt.exceptions.ExecutionError] giving
the reason the first lost evaluation gave. It is not absorbed as a `NaN`,
because a result computed over the workers that survived is indistinguishable
from one computed over the whole ensemble.

| What you see | Most likely cause |
| --- | --- |
| `result.results` is `None`, but `exit_code` is `FINISHED` | No evaluation satisfied the constraints to within `constraint_tolerance` (default `1e-10`; bounds and linear constraints count too), so there is no best feasible result to return. The evaluations are still in the handlers. Raise the tolerance, or check that the constraints can be satisfied at all. |
| `TOO_FEW_REALIZATIONS` although only one realization failed | [`realization_min_success`](../optimizer_setup/configuration_sections.md#realizations) defaults to all of them. Lower it. |
| The optimization stopped long before `max_functions`, with `FINISHED` | The backend ended it: `max_iterations`, `convergence_tolerance`, or its own convergence test. Those are set in the `backend` section; see [Limiting the length of an optimization](../optimizer_setup/optimizer.md#limiting-the-length-of-a-run). |
| `ExecutionError` part-way through an optimization | The worker pool failed, not your objective. The message names the reason the first lost evaluation gave. |
| Numbers do not match what you configured | Results carry the configured values and the optimizer's scaled ones side by side; see [Scaling of results](../results/results.md#scaling-of-results). |

## Your evaluation function

**Outside your process, your objective works on a copy.** On a
[process pool](../running/parallel.md#process-pool) or an HPC pool,
your evaluation function is sent to a worker together with the data it uses. Anything
it writes there — a global, a cache, a list it appends to — is thrown away when
the worker finishes. Return what you need instead; see
[Handlers and separate processes](../results/handlers.md#handlers-and-separate-processes).

**Several optimizations may call your objective at the same time.**
Optimizations started together with [`run`][ropt.run] and a pool
run concurrently. Given a pool, their evaluations go there; given none, each
optimization evaluates on the worker of `run` that runs it, so your
evaluation function is called from several threads at once and has to tolerate
that. See [Running Multiple Optimizations](../running/multiple_runs.md).

**Metadata is copied with `copy.deepcopy`.** Every result gets its own copy of
the `metadata` you passed, so keep it to plain data — numbers, strings, lists,
arrays. A lock, an open file, or a database connection cannot be copied and
raises a `TypeError` in the middle of the optimization.

**An exception from your objective is not turned into a failed realization.** It
is re-raised from the `optimize` call, whichever pool the evaluation ran on, so
a bug surfaces as the exception you wrote rather than as a missing value. Return
`float("nan")` or `numpy.nan` where a realization that could not produce a value
should be tolerated; see [When an evaluation
fails](../running/parallel.md#when-an-evaluation-fails).

| What you see | Most likely cause |
| --- | --- |
| A global or a cache your objective writes to is never updated | The objective ran in a worker process, on a copy. Return the value as [metadata](../running/running.md#attaching-metadata) instead. |
| An objective that works alone misbehaves when several optimizations run at once | It is being called from several threads at once. Remove the shared mutable state, or guard it with a lock of its own. |
| `TypeError: cannot pickle ...` part-way through an optimization | Something in `metadata` cannot be deep-copied. |
| `ExecutionError` part-way through an optimization with a pool | A worker or job broke, rather than your objective; see [When an evaluation fails](../running/parallel.md#when-an-evaluation-fails). |

## Pools, stopping and processes

**An optimization evaluates with the pool it was given, and no other.**
Nothing is picked up from the surrounding code. An optimization started with
[`optimize`][ropt.optimize] without a pool evaluates inside your own program, on
the thread that called it — even if a pool exists next to it.

**Threads cannot be interrupted.** Evaluations with a thread pool run to
completion even after Ctrl-C, because Python cannot interrupt a thread from
outside. An evaluation that checks [`aborting`][ropt.aborting], or runs its
external programs with [`run_program`][ropt.run_program], stops early; any other
evaluation that may run long and has to be interruptible belongs on a process
pool or an HPC pool; see
[Stopping an optimization](../running/parallel.md#stopping-a-run).

**A process pool re-imports your script.** Its workers are started with
`spawn`, so each of them imports the file your program was started from, and
everything at module level runs again in every worker. That is why the entry
point has to sit behind `if __name__ == "__main__":`. Without the guard the
workers try to start workers of their own, and building the pool raises.

**A pool holds its workers until its session closes.** Open the session inside
the loop or function that uses it rather than around the whole program; see
[Releasing a pool early](../running/parallel.md#how-many-workers).

| What you see | Most likely cause |
| --- | --- |
| Ctrl-C appears to do nothing | An imported package may have changed a process-wide signal setting; affected `polars` versions do this when installed. Check [Keyboard Interrupts](keyboard_interrupt.md) for how to test and use [`restore_keyboard_interrupt`][ropt.utils.restore_keyboard_interrupt] if needed. |
| The program will not exit after Ctrl-C | Evaluations with a thread pool are still running and cannot be interrupted. Check [`aborting`][ropt.aborting] in them, or start their programs with [`run_program`][ropt.run_program]. |
| Pages of `multiprocessing` tracebacks, ending in `ExecutionError: Could not start worker processes` | The script has no `if __name__ == "__main__":` guard, so every worker re-ran it from the top. |
| A process pool keeps hundreds of megabytes resident | Each worker is a separate interpreter. Close the session that built the pool when the optimizations that use it are done. |
| `WorkflowError` about a pool released when its session closed | The pool outlived its `with session()` block. Build a new one inside an open session. |
| More workers made everything slower | `numpy` and similar libraries already use every core. Set `OMP_NUM_THREADS=1` and let the pool provide the parallelism; see [Which pool should I use?](../running/parallel.md#which-pool). |
| Simulators keep running after the optimization stopped | They were not started with [`run_program`][ropt.run_program], which terminates a program and everything it started on an abort; see [Running External Programs](../running/programs.md). |
| A cluster directive has no effect | The submission script never mentions that variable, or the value was clamped to the queue's limit; see [Running on an HPC cluster](../running/parallel.md#running-on-an-hpc-cluster). |

## Running many at once

**Optimizations started together really do run at once.** A handler given to all
of them takes a lock around every event it handles, so it collects them safely,
but a single result callback does not: it is called by several threads at the
same time. Keep such a callback free of shared state, or give each optimization
its own.

**A slow shared handler holds up every optimization.** The optimization that
emitted a result waits until the handler has finished with it, and the next one
waits for the lock. Keep a shared handler cheap.

**A handler must not start an optimization that reaches it again.** The handler
holds its own lock while it runs. A nested [`optimize`][ropt.optimize] emits on
that same thread and raises; a nested optimization started with
[`run`][ropt.run] and a pool emits on a worker of that pool, and blocks. See [Result
Handlers](../results/handlers.md#sharing-a-handler-across-concurrent-runs).

**Not every optimizer can run beside another.** A backend that needs its own
working directory, writes to a fixed file name, or keeps state inside its
library cannot run concurrently in one process; select it as
[`external/...`](../optimizer_setup/optimizer.md#external-backend) to give it a
process of its own. Optimizer output capture is likewise for one optimization at
a time.

| What you see | Most likely cause |
| --- | --- |
| Results from a shared result callback are jumbled or lost | The callback is called from the thread of every optimization at once. Collect the results in a [handler](../results/handlers.md#sharing-a-handler-across-concurrent-runs) instead, or use one callback per optimization. |
| An optimization started from a handler raises `WorkflowError` about the call stack, or hangs | Its handler list reaches a handler that is already running; see [Result Handlers](../results/handlers.md#sharing-a-handler-across-concurrent-runs). |
| Several optimizations are slower than expected while the workers sit idle | A shared handler is serializing them. Make it cheaper. |
| A second concurrent optimization raises `WorkflowError` about output capture | Only one optimization at a time may set `stdout` or `stderr`; see [Running Multiple Optimizations](../running/multiple_runs.md). |

## Configuration

**Optimizations are reproducible by default.** The
[seed](../optimizer_setup/configuration_sections.md#variable-perturbations) is fixed at
`1` unless you set it, so repeating an optimization reproduces its perturbations
exactly.
Change the seed when you deliberately want an independent repetition of the same
problem.

**Label tuples are checked at export, not at the start.** A tuple in
[`names`](../optimizer_setup/configuration_sections.md#names) that is not as long as the
axis it labels lets the optimization finish normally; the mismatch surfaces later,
as an
error from pandas or polars when a result is turned into a table.

| What you see | Most likely cause |
| --- | --- |
| `RuntimeError: Auto-scaling of the objectives failed to estimate a scale factor` | The first batch averaged to zero, or to a value that is not finite, so no scale could be derived. Set the scales yourself, or start somewhere else. |
| Two optimizations that should agree give different gradients | Quasi-random samplers (`sobol`, `halton`, `lhs`) are seeded in the order the [samplers](../optimizer_setup/configuration_sections.md#samplers) are defined, so reordering that section changes the perturbations. |
