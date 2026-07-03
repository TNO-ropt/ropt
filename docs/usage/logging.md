# Logging

`ropt` uses Python's standard [`logging`](https://docs.python.org/3/library/logging.html)
module to report what is happening during an optimization workflow. By default
it produces **no output at all** — a `NullHandler` is installed on the `ropt`
logger so that log records are silently discarded unless an application
explicitly enables them.

## Logger hierarchy

Every module in `ropt` creates its own logger whose name is derived from the
public package path:

```
ropt
├── ropt.workflow
│   └── ropt.workflow.compute_steps    ← OptimizationStep, EvaluationStep
├── ropt.core                          ← EnsembleOptimizer, EnsembleEvaluator
└── ropt.plugins
    └── ropt.plugins.manager           ← PluginManager
```

This means you can enable logging for the entire library by configuring the
`ropt` logger, or limit output to a sub-tree such as `ropt.core`.

## What is logged

### `INFO` — workflow milestones and batch statistics

These messages tell you what the optimization is doing at a human level.

| Source | Example message |
|--------|----------------|
| `OptimizationStep`  | `Starting optimization` |
| `OptimizationStep`  | `Optimization finished: Optimization finished successfully` |
| `EvaluationStep`    | `Starting evaluation` |
| `EvaluationStep`    | `Evaluation finished: Ensemble evaluation finished successfully` |
| `EnsembleOptimizer` | `Stopping: Maximum number of function evaluations reached (500)` |
| `EnsembleOptimizer` | `Stopping: Too few realizations were evaluated successfully` |
| `EnsembleEvaluator` | `Function evaluation: 9/10 realizations succeeded` |
| `EnsembleEvaluator` | `Gradient evaluation: 8/10 realizations succeeded` |

The batch statistics after each evaluation are especially useful for monitoring
realization failures without having to write a custom event handler.

### `DEBUG` — per-callback trace

These messages are emitted once per optimizer callback invocation and are
useful for detailed diagnostics. They can be **verbose**: a gradient-based
optimizer typically calls the evaluation callback once for functions and once
for gradients per iteration.

| Source | Example message |
|--------|----------------|
| `EnsembleOptimizer` | `Optimizer callback: requesting functions` |
| `EnsembleOptimizer` | `Optimizer callback: requesting gradients` |
| `EnsembleOptimizer` | `Optimizer callback: requesting functions and gradients` |
| `PluginManager`     | `Registering plugin: backend/scipy` |

## Enabling logging

### Minimal — see everything from `ropt`

```python
import logging

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
```

This outputs `INFO` and above from all loggers, including `ropt`. Example
output during a short optimization run:

```
ropt.workflow.compute_steps - INFO - Starting optimization
ropt.core - INFO - Function evaluation: 10/10 realizations succeeded
ropt.core - INFO - Gradient evaluation: 10/10 realizations succeeded
ropt.core - INFO - Function evaluation: 10/10 realizations succeeded
ropt.core - INFO - Gradient evaluation: 9/10 realizations succeeded
...
ropt.core - INFO - Stopping: Maximum number of function evaluations reached (200)
ropt.workflow.compute_steps - INFO - Optimization finished: Maximum number of function evaluations reached (200)
```

### High-level only — workflow messages without core detail

Because `ropt.core` covers both stopping conditions and per-batch statistics,
you cannot suppress one without the other by logger name alone. To see only
workflow start/stop messages, enable `INFO` on `ropt.workflow.compute_steps`
and leave `ropt.core` at `WARNING`:

```python
import logging

logging.basicConfig(level=logging.WARNING)  # silence everything by default

logging.getLogger("ropt.workflow.compute_steps").setLevel(logging.INFO)
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
