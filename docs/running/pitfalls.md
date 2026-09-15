# Common Pitfalls

`ropt` tries to keep a run going rather than stop it at the first sign of
trouble. That is usually what you want, but it means a run can end without a
result to return, and without raising an error to say so. This page collects the
behaviours that most often cause confusion.

Read it once from top to bottom, then come back to the
[table below](#when-something-looks-wrong) when a run does something you did not
expect.

## What to keep in mind

**A run uses the pool you hand it, and no other.** Nothing is picked up from the
surrounding code. A run given no `pool=` evaluates in-process, on the thread that
called it — even if a session is open next to it.

**Outside your process, your objective works on a copy.** On a
[`process_pool`](../getting_started/execution.md#process-pool), a
[`local_pool`](../getting_started/execution.md#local-pool) or an `hpc_pool`, your
evaluation function is sent to a worker together with the data it uses. Anything
it writes there — a global, a cache, a list it appends to — is thrown away when
the worker finishes. Return what you need instead; see
[Handlers and the process boundary](handlers.md#handlers-and-the-process-boundary).

**`NaN` means "this realization failed", and one failure is already too many.**
[`realization_min_success`](../optimizer_setup/configuration.md#realizations)
defaults to *all* realizations, so a single `NaN` ends the run with
`TOO_FEW_REALIZATIONS`. If some realizations are allowed to fail, say so:

```python
config = {
    "variables": {"variable_count": 3, "perturbation_magnitudes": 1e-6},
    "realizations": {"weights": [1.0] * 10, "realization_min_success": 8},
}
```

**The returned result is the best *feasible* one.**
[`optimize`][ropt.simple.optimize] returns the best result that satisfies every
constraint to within `constraint_tolerance`, which defaults to `1e-10` and
applies to bounds and linear constraints as well as nonlinear ones. If no
evaluation ever clears that bar, the run ends normally with `result.variables`
set to `None`. The evaluations themselves are not lost: every result, feasible
or not, still reaches the [handlers](handlers.md) attached to the run.

**The return value is a summary, not the record of the run.** It holds a single
result, the best feasible evaluation. Most analysis works from the whole history
instead: attach a [`HistoryHandler`][ropt.simple.HistoryHandler] or a
[`DataFrameHandler`][ropt.simple.DataFrameHandler] to collect every result as it
arrives. That history is also what remains when there is no best result to
return.

**Not everything that goes wrong raises.** Check `exit_code` before using a
result, and remember that `TOO_FEW_REALIZATIONS` and `EXECUTOR_STOPPED` leave
every field `None`. See
[When something goes wrong](running.md#when-something-goes-wrong).

**Some reasons are only ever logged.** When the machinery itself fails — a
worker process is killed, a cluster job never writes its result — the affected
evaluations are recorded as `NaN`. The optimizer sees only the `NaN`, so the
reason appears once, as a `WARNING` from the `ropt.components.evaluators`
logger, and nowhere else. Turn logging on before investigating a run that ended
in `TOO_FEW_REALIZATIONS`; see [Logging](../utilities/logging.md).

**`optimize_many` really does run everything at once.** A handler must be in a
[shared group](handlers.md#sharing-a-handler-across-concurrent-runs), and a
single `report=` callback given to all the runs is called by several threads at
the same time. Keep such a callback free of shared state, or give each run its
own.

**Metadata is copied with `copy.deepcopy`.** Every result gets its own copy of
the `metadata` you passed, so keep it to plain data — numbers, strings, lists,
arrays. A lock, an open file, or a database connection cannot be copied and
raises a `TypeError` in the middle of the run.

**Runs are reproducible by default.** The
[seed](../optimizer_setup/configuration.md#variable-perturbations) is fixed at
`1` unless you set it, so repeating a run reproduces its perturbations exactly.
Change the seed when you deliberately want an independent repetition of the same
problem.

**Threads cannot be hurried.** Evaluations on a `thread_pool` run to completion
even after Ctrl-C, because Python cannot interrupt a thread from outside. If an
evaluation may run long and has to be interruptible, put it on a `local_pool` or
an `hpc_pool`; see [Stopping a run](../getting_started/execution.md#stopping-a-run).

## When something looks wrong

| What you see | Most likely cause |
| --- | --- |
| `result.variables` is `None`, but `exit_code` is `OPTIMIZER_FINISHED` | No evaluation satisfied the constraints to within `constraint_tolerance` (default `1e-10`; bounds and linear constraints count too), so there is no best feasible result to return. The evaluations are still in the handlers. Raise the tolerance, or check that the constraints can be satisfied at all. |
| `TOO_FEW_REALIZATIONS` although only one realization failed | [`realization_min_success`](../optimizer_setup/configuration.md#realizations) defaults to all of them. Lower it. |
| `TOO_FEW_REALIZATIONS` and nothing says why | The failures came from the machinery, not from your objective. The reason is logged at `WARNING` by `ropt.components.evaluators`; enable [logging](../utilities/logging.md). |
| A global or a cache your objective writes to is never updated | The objective ran in a worker process, on a copy. Return the value as [metadata](running.md#attaching-metadata) instead. |
| A handler collects nothing from `optimize_many` | A plain handler belongs to one run. Put it in a [shared group](handlers.md#sharing-a-handler-across-concurrent-runs). |
| Results from a shared `report=` callback are jumbled or lost | The callback is called from every run's thread at once. Use a shared group, or one callback per run. |
| `TypeError: cannot pickle ...` part-way through a run | Something in `metadata` cannot be deep-copied. |
| `RuntimeError: Auto-scaling of the objectives failed to estimate a scale factor` | The first batch averaged to zero, or to a value that is not finite, so no scale could be derived. Set the scales yourself, or start somewhere else. |
| Ctrl-C appears to do nothing | A process-wide signal setting, changed by an imported package. Call [`restore_keyboard_interrupt`][ropt.utils.restore_keyboard_interrupt]; see [Keyboard Interrupts](../utilities/keyboard_interrupt.md). |
| The program will not exit after Ctrl-C | Evaluations on a `thread_pool` are still running and cannot be interrupted. |
| More workers made everything slower | `numpy` and friends already use every core. Set `OMP_NUM_THREADS=1` and let the pool provide the parallelism; see [Running in Parallel](../getting_started/execution.md#which-one-should-i-use). |
| Simulators keep running after the run stopped | A `process_pool` kills only its own workers. Use a [`local_pool`](../getting_started/execution.md#local-pool), which signals the whole process group. |
| A cluster directive has no effect | The submission script never mentions that variable, or the value was clamped to the queue's limit; see [Running on an HPC cluster](parallel.md#running-on-an-hpc-cluster). |
| Two runs that should agree give different gradients | Quasi-random samplers (`sobol`, `halton`, `lhs`) are seeded in the order the [samplers](../optimizer_setup/configuration.md#samplers) are defined, so reordering that section changes the perturbations. |
| Numbers do not match what you configured | Results carry the configured values and the optimizer's scaled ones side by side; see [Scaling of results](../optimizer_setup/results.md#scaling-of-results). |

## Where to next

- The basics of a single run: [Running Optimizations](running.md).
- Pools, many runs, and what stopping does:
  [Parallel Execution and Many Runs](parallel.md) and
  [Running in Parallel](../getting_started/execution.md).
- The same failure model at the low level, in more detail:
  [Error handling](../workflows/parallel.md#error-handling).
