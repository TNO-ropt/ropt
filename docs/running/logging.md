# Logging

`ropt` reports its progress through Python's standard
[`logging`](https://docs.python.org/3/library/logging.html) module. Switching it
on takes one line and needs no code of your own. To *collect* what a run
produces — to keep results, tabulate them, or stop a run early — use
[result handlers](handlers.md) instead.

## Turning it on

Put this before the run:

```python
import logging

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
```

A short optimization over four realizations then reports:

```
ropt.components.compute_steps - INFO - Starting optimization
ropt.core - INFO - Function evaluation: 4/4 realizations succeeded
ropt.components.event_handlers - INFO - New best objective: 2
ropt.core - INFO - Gradient evaluation: 4/4 realizations succeeded
ropt.core - INFO - Function evaluation: 4/4 realizations succeeded
ropt.components.event_handlers - INFO - New best objective: 1.99886
ropt.core - INFO - Function evaluation: 4/4 realizations succeeded
ropt.components.event_handlers - INFO - New best objective: 1.50006
ropt.core - INFO - Stopping: Maximum number of function evaluations reached (3)
ropt.components.compute_steps - INFO - Optimization finished: MAX_FUNCTIONS_REACHED
```

Without that line `ropt` prints nothing at all: its output is discarded until an
application adds a destination for it.

## Reading the output

**The per-batch counts** appear one per batch of evaluations, and `4/4` means an
ensemble of 4 evaluations came back with a value. While these keep arriving, the
run is alive and calling your objective. A `3/4` after a run of `4/4` indicates a
failed realization; how many failures a run tolerates is set by
[`realization_min_success`](../optimizer_setup/configuration_sections.md#realizations).

**A new best objective** is reported each time the run improves on what it had.

**The reason for stopping** is stated before the run ends, and the closing line
names the [`ExitCode`][ropt.enums.ExitCode] it finished with — the same one
`result.exit_code` reports. See
[When something goes wrong](running.md#when-something-goes-wrong) for what each
code means.

The name at the start of each line identifies the part of `ropt` that produced
it. That is what a bug report needs; leave
`%(name)s` out of the format string if it is in the way.

## Warnings only

`INFO` is one line per batch, which adds up on a long run. Setting the level to
`WARNING` instead reports nothing while the run proceeds normally, and reports
only events that did not stop it:

```python
logging.basicConfig(level=logging.WARNING)
```

Typical examples are a scheduler that had to be queried twice, a cluster job that
could not be cancelled and may still be running, a working directory kept behind
so you can see what a failed job left, and evaluations that must finish before
your program is allowed to exit. None of these raise, and most are reported
nowhere else.

A failure that stops a run does not need logging to be seen: it either raises,
or is named by the exit code the run ends with. See
[Troubleshooting](../troubleshooting/index.md).

## Sending the output somewhere else

`logging.basicConfig()` writes to the console. To send `ropt`'s output somewhere
of its own — a file, a window, a queue — give the `ropt` logger a destination and
stop its output from reaching the rest of your program's logging:

```python
import logging

log_file = logging.FileHandler("optimization.log")
log_file.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

ropt_logger = logging.getLogger("ropt")
ropt_logger.setLevel(logging.INFO)
ropt_logger.addHandler(log_file)
ropt_logger.propagate = False
```

The last line is what separates `ropt` from the rest of your program. Without
it, the output reaches this file *and* whatever the rest of your application
does with logging — so if you also called `logging.basicConfig()`, every line
appears twice.

## While the optimizer is running { #logging-during-an-optimization }

Setting [`stdout` or
`stderr`](../optimizer_setup/configuration_sections.md#optimizer) sends the
optimizer's own output to a file. That capture is scoped to a period of time
rather than to a source, so other output produced during that period could end
up in the file as well. Almost no log output does: `ropt` lifts the capture
around every evaluation, and a console destination created before the run keeps
writing to the stream it was given.

The exception is an optimizer that prints from compiled code. `ropt` then
redirects the process's output streams as well, and at that level console
logging *is* caught: its lines land in the capture file along with the
optimizer's own. Among the SciPy methods this applies to `tnc`. Giving `ropt` a
destination of its own, as above, keeps its output out of the file in that case
too.

Python warnings go the other way round. A `warnings.warn` raised while the
optimizer is working *is* captured, which is usually what you want, since it
generally comes from the optimizer.

??? info "Narrowing it down further"

    Every line comes from a logger named after the part of `ropt` that produced
    it, and all of them sit under `ropt`, which is why setting the level there
    covers the library. Setting it on one of the names below covers that part
    alone, which is the quickest way to follow a single mechanism: put
    `ropt.components.executors` at `DEBUG` to watch jobs being submitted
    without the per-batch traffic.

    | Logger | Emits |
    | --- | --- |
    | `ropt.components.compute_steps` | the start and finish of a run, and the configuration it started from |
    | `ropt.core` | per-batch realization counts, stopping conditions, optimizer callbacks, and reuse of cached results |
    | `ropt.components.executors` | executor start-up, job submission and cancellation, retained working directories, and the thread-pool drain warning |
    | `ropt.components.evaluators` | work-item dispatch |
    | `ropt.components.event_handlers` | each new best objective, and any handler that raised |
    | `ropt.backend.scipy`, `ropt.backend.external` | the method in use, and the external subprocess lifecycle |
    | `ropt.plugins` | plugin registration |

    `DEBUG` traces the mechanism: one line per optimizer callback, per
    dispatched batch and per cluster job, plus the configuration the run
    started from. It is verbose — a gradient-based method requests functions
    and gradients separately on most iterations, and each request is a line —
    but it is what to attach to a bug report. See
    [Optimization Workflows](../advanced/workflows.md) for what these
    components are.
