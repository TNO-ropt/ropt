# Logging

`ropt` reports what a run is doing through Python's standard
[`logging`](https://docs.python.org/3/library/logging.html) module. It produces
**no output at all** until an application asks for it: a `NullHandler` on the
`ropt` logger discards every record until you attach a handler of your own.

What logging gives you is a *trace* — a readable account of a run as it
happens. It is not a way to get at results: to collect them, tabulate them, or
stop a run early, use [result handlers](../running/handlers.md) instead.

## Turning it on

One line, before the run:

```python
import logging

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
```

A short optimization over four realizations then reports:

```
ropt.components.compute_steps - INFO - Starting optimization
ropt.core - INFO - Function evaluation: 4/4 realizations succeeded
ropt.components.event_handlers - INFO - New best objective: 3
ropt.core - INFO - Gradient evaluation: 3/4 realizations succeeded
ropt.core - INFO - Function evaluation: 4/4 realizations succeeded
ropt.components.event_handlers - INFO - New best objective: 2.99998
ropt.components.compute_steps - INFO - Optimization finished: OPTIMIZER_FINISHED
```

Every record comes from a logger under `ropt`, named after the part of the
library that emitted it.

## What each level carries

**`INFO` follows the run.** A line when the run starts and another when it
ends, carrying the [`ExitCode`][ropt.enums.ExitCode] name; one line per batch
giving how many realizations produced a value; and a line each time the best
objective improves. A run that stops on a limit you set says so before it
finishes:

```
ropt.core - INFO - Stopping: Maximum number of function evaluations reached (200)
```

The per-batch counts are the most useful of these. A `3/4` after a line that
said `4/4` tells you a realization failed, without writing a handler to find
out.

**`WARNING` reports trouble the run survived.** A worker process that died, a
cluster job that never produced a result, a scheduler query that had to be
retried, a working directory kept because something in it failed.

One property of these is worth knowing. When the machinery itself fails, the
affected evaluations are recorded as `NaN`, and the optimizer sees nothing but
the `NaN`. The reason is stated once, in a warning, and nowhere else — so a run
that ends in `TOO_FEW_REALIZATIONS` either explains itself here or not at all.

**`DEBUG` traces the mechanism.** One record per optimizer callback, per
dispatched batch and per cluster job, plus the configuration the run started
from. It is verbose: a gradient-based method asks for functions and gradients
separately on most iterations, and each request is a line.

## Choosing how much you see

Setting the level on `ropt` covers the whole library. The loggers beneath it
narrow that down, and two of them account for most of what appears at `INFO`:
`ropt.components.compute_steps` emits the start and finish of a run, while
`ropt.core` emits the per-batch counts and the stopping reason.

Those two cannot be separated by level alone, because `ropt.core` carries both.
To keep the milestones without a line per batch, silence everything and raise
only the step logger:

```python
import logging

logging.basicConfig(level=logging.WARNING)
logging.getLogger("ropt.components.compute_steps").setLevel(logging.INFO)
```

Adding `logging.getLogger("ropt.core").setLevel(logging.INFO)` brings the batch
counts and the stopping conditions back.

## Keeping `ropt`'s records separate

Records travel up from each logger to its parent until they reach the root
logger, which is where `logging.basicConfig()` installs its handler. That is
why the single line above is enough to see `ropt` output — and why attaching a
handler to `ropt` *as well* prints everything twice.

Setting `propagate = False` on the `ropt` logger stops records there, so they
reach only the handlers you attach to it:

```python
import logging

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(levelname)s %(message)s"))

log_file = logging.FileHandler("optimization.log")
log_file.setLevel(logging.DEBUG)
log_file.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))

ropt_logger = logging.getLogger("ropt")
ropt_logger.setLevel(logging.DEBUG)
ropt_logger.addHandler(console)
ropt_logger.addHandler(log_file)
ropt_logger.propagate = False
```

Use this whenever `ropt`'s output should go somewhere of its own — a file, a
widget, a queue — independently of what the rest of the application does with
logging.

## Logging while the optimizer is running { #logging-during-an-optimization }

Setting [`stdout` or
`stderr`](../optimizer_setup/configuration_sections.md#optimizer) sends the
optimizer's own output to a file. That capture is scoped to a period of time
rather than to a source, so it is fair to ask what else ends up in the file.
For log records, almost nothing does.

Two things keep them out. `ropt` lifts the capture for the whole evaluation
phase, so everything logged around a batch — and everything your own code
prints there — goes where it normally goes. And a `StreamHandler` writes to the
stream object it was given when it was created, so rebinding `sys.stderr`
underneath it, which is what the capture does, does not reach it.

The exception is an optimizer that prints from compiled code. `ropt` then
redirects file descriptors 1 and 2 as well, and at that level a console handler
*is* caught: its records land in the capture file along with the optimizer's
output. Among the SciPy methods this applies to `tnc`.

Giving the `ropt` logger a destination of its own — a file, a socket, a queue —
and setting `propagate = False`, as above, keeps its records out of the capture
file in that case too.

Python warnings go the other way: `warnings.warn` looks up `sys.stderr` when it
fires, so a warning raised while the optimizer is working is captured. That is
usually what you want, since it generally comes from the optimizer.

??? info "Which part of `ropt` emits what"

    A logger is named after the public package path of the module that emits
    the record, so the names map onto the component API:

    | Logger | Emits |
    | --- | --- |
    | `ropt.components.compute_steps` | the configuration a step started from, and its start and finish |
    | `ropt.core` | optimizer callbacks, per-batch realization counts, stopping conditions, and reuse of cached function results |
    | `ropt.components.evaluators` | work-item dispatch, cache statistics, and the reason evaluations were recorded as failed |
    | `ropt.components.executors` | executor start-up, job submission and cancellation, retention of a working directory, and the thread-pool drain warning |
    | `ropt.components.event_handlers` | each new best objective, and an event handler that raised |
    | `ropt.backend.scipy`, `ropt.backend.external` | the method in use, and the external subprocess lifecycle |
    | `ropt.plugins` | plugin registration |

    Scoping to one sub-tree is the quickest way to watch a single mechanism:
    put `ropt.components.executors` at `DEBUG` to follow job submission without
    the per-batch traffic. See
    [Optimization Workflows](../advanced/workflows.md) for what these
    components are.
