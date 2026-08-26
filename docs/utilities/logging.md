# Logging

`ropt` uses Python's standard [`logging`](https://docs.python.org/3/library/logging.html)
module to report what is happening during an optimization workflow. By default
it produces **no output at all** — a `NullHandler` is installed on the `ropt`
logger so that log records are silently discarded unless an application
explicitly enables them.

Logging gives a human-readable *trace* of a run. To *react* to results
programmatically — collect them, tabulate them, or stop early — use
[result handlers](../running/handlers.md#result-handlers) instead.

## Logger hierarchy

Every module in `ropt` creates its own logger whose name is derived from the
public package path:

```
ropt
├── ropt.backend                         ← backend-specific messages (SciPy, external)
├── ropt.components
│   ├── ropt.components.compute_steps    ← OptimizationStep, EvaluationStep
│   ├── ropt.components.evaluators       ← CachedEvaluator, ParallelEvaluator
│   ├── ropt.components.event_handlers   ← ResultsHandler
│   └── ropt.components.executors        ← Threading/Multiprocessing/HPCExecutor
├── ropt.core                            ← EnsembleOptimizer, EnsembleEvaluator
└── ropt.plugins
    └── ropt.plugins.manager             ← PluginManager
```

This means you can enable logging for the entire library by configuring the
`ropt` logger, or limit output to a sub-tree such as `ropt.core` or
`ropt.components.executors` (useful when debugging HPC job submission without
the noise of per-batch statistics).

## What is logged

### `INFO` — workflow milestones and batch statistics

These messages tell you what the optimization is doing at a human level.

| Source | Example message |
|--------|----------------|
| `OptimizationStep`  | `Starting optimization` |
| `OptimizationStep`  | `Optimization finished: OPTIMIZER_FINISHED` (the [`ExitCode`][ropt.enums.ExitCode] name) |
| `EvaluationStep`    | `Starting evaluation` |
| `EvaluationStep`    | `Evaluation finished` |
| `EnsembleOptimizer` | `Stopping: Maximum number of function evaluations reached (500)` |
| `EnsembleOptimizer` | `Stopping: Maximum number of evaluation batches reached (50)` |
| `EnsembleEvaluator` | `Function evaluation: 9/10 realizations succeeded` |
| `EnsembleEvaluator` | `Gradient evaluation: 8/10 realizations succeeded` |
| `ResultsHandler`    | `New best objective: 1.23456` |
| `HPCExecutor`       | `Starting HPC executor (4 max workers, 1.0s poll interval)` |
| `external` (backend) | `Starting external optimization in subprocess` |

The batch statistics after each evaluation are especially useful for monitoring
realization failures without having to write a custom event handler. Note that
a run stopped by [`TooFewRealizations`][ropt.exceptions.TooFewRealizations]
(exit code `TOO_FEW_REALIZATIONS`) logs no separate "stopping" message of its
own — only the final `Optimization finished: TOO_FEW_REALIZATIONS` line.

### `WARNING` — recoverable problems

These signal something went wrong that `ropt` could recover from (a retry, a
dropped job, a lost worker) — usually worth surfacing even when you otherwise
run at `INFO` or above.

| Source | Example message |
|--------|----------------|
| `HPCExecutor`               | `HPC work item <id> failed: output file never appeared` |
| `HPCExecutor`               | `HPC work item <id> failed: no valid result after 30 retries` |
| `HPCExecutor`               | `Querying the HPC scheduler failed (2/31): <error>` |
| `HPCExecutor`               | `Could not cancel HPC job <id> (job id: <job>): <error>` |
| `ParallelEvaluator`         | `Recording 1 evaluation(s) as failed: <reason>` |
| `ProcessExecutor`   | `Worker process pool broken; work item result lost` |
| `external` (backend)        | `External backend subprocess died unexpectedly (exit code <code>)` |

The `ParallelEvaluator` message is the only place an infrastructure failure
states its reason: the optimizer sees nothing but `numpy.nan`, so a run that
ends in `TOO_FEW_REALIZATIONS` explains itself here and nowhere else.

### `DEBUG` — per-callback and per-task trace

These messages are emitted once per optimizer callback invocation, or once per
dispatched task, and are useful for detailed diagnostics. They can be
**verbose**: a gradient-based optimizer typically calls the evaluation
callback once for functions and once for gradients per iteration.

| Source | Example message |
|--------|----------------|
| `EnsembleOptimizer`         | `Optimizer callback: requesting functions` |
| `EnsembleOptimizer`         | `Optimizer callback: requesting gradients` |
| `EnsembleOptimizer`         | `Optimizer callback: requesting functions and gradients` |
| `PluginManager`             | `Registering plugin: backend/scipy` |
| `scipy` (backend)           | `Using SciPy optimizer: SLSQP` |
| `ThreadExecutor`            | `Starting thread executor with 4 worker(s)` |
| `ProcessExecutor`           | `Starting process executor with 4 worker(s)` |
| `HPCExecutor`               | `Submitted HPC job <id> (job id: <job>)` |
| `ParallelEvaluator`         | `Dispatching 10 work item(s) to executor` |
| `CachedEvaluator`           | `Cache: 4/10 evaluations served from cache` |

## Enabling logging

### Minimal — see everything from `ropt`

```python
import logging

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
```

This outputs `INFO` and above from all loggers, including `ropt`. Example
output during a short optimization run:

```
ropt.components.compute_steps - INFO - Starting optimization
ropt.core - INFO - Function evaluation: 10/10 realizations succeeded
ropt.core - INFO - Gradient evaluation: 10/10 realizations succeeded
ropt.core - INFO - Function evaluation: 10/10 realizations succeeded
ropt.core - INFO - Gradient evaluation: 9/10 realizations succeeded
...
ropt.core - INFO - Stopping: Maximum number of function evaluations reached (200)
ropt.components.compute_steps - INFO - Optimization finished: Maximum number of function evaluations reached (200)
```

### High-level only — workflow messages without core detail

Because `ropt.core` covers both stopping conditions and per-batch statistics,
you cannot suppress one without the other by logger name alone. To see only
workflow start/stop messages, enable `INFO` on `ropt.components.compute_steps`
and leave `ropt.core` at `WARNING`:

```python
import logging

logging.basicConfig(level=logging.WARNING)  # silence everything by default

logging.getLogger("ropt.components.compute_steps").setLevel(logging.INFO)
# ropt.core stays at WARNING → no batch statistics and no stopping conditions
```

To also include stopping conditions and batch statistics, add `ropt.core`:

```python
logging.getLogger("ropt.core").setLevel(logging.INFO)
```

### Verbose — include per-callback trace

```python
import logging

logging.basicConfig(level=logging.DEBUG, format="%(name)s - %(levelname)s - %(message)s")
```

### `ropt` only, leaving other loggers at their current level

By default, every logger passes its records up to its parent until they reach
the **root logger**. This is called *propagation*. If the root logger already
has a handler — for example because the application called
`logging.basicConfig()` — then adding a handler to `ropt` as well would send
each `ropt` record through *two* handlers and print it twice.

Setting `propagate = False` on the `ropt` logger cuts the chain: records from
`ropt` and all its children are handled exclusively by the handlers you attach
to `ropt` and never reach the root.

```python
import logging

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))

ropt_logger = logging.getLogger("ropt")
ropt_logger.setLevel(logging.INFO)
ropt_logger.addHandler(handler)
ropt_logger.propagate = False  # records stop here; root logger is not involved
```

Use this pattern whenever you want `ropt` output to go to a specific
destination (a file, a widget, a queue) independently of whatever the rest of
the application is doing with logging.

## Integration with log file and console simultaneously

```python
import logging

# Console: INFO and above
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(levelname)s %(message)s"))

# File: everything including DEBUG
file_handler = logging.FileHandler("optimization.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
)

ropt_logger = logging.getLogger("ropt")
ropt_logger.setLevel(logging.DEBUG)
ropt_logger.addHandler(console)
ropt_logger.addHandler(file_handler)
ropt_logger.propagate = False
```

## Where to next

- React to results programmatically instead of just tracing them:
  [Result Handlers](../running/handlers.md).
- HPC job submission, polling, and retries in depth:
  [Parallel Evaluation](../workflows/parallel.md).
- Query installed plugins at runtime: [Plugin Discovery](plugin_discovery.md).
