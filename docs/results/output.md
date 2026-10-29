# Getting Output

An optimization can return its best result, report evaluations as they finish,
store results for later analysis, and write progress messages. This page helps
you choose how to get the output you need; the linked pages explain each option.

## What was the best result, and why did the optimization stop?

[`optimize`][ropt.optimize] returns an `OptimizationResult` with the best
feasible result and an exit code. The result is `None` if the optimization did
not produce a feasible point. Read the exit code to distinguish convergence,
limits, a callback stop, and insufficient successful evaluations. An abort
raises an exception instead.

See [The result](../running/running.md#the-result),
[Exit Codes](exit_codes.md), and
[Working with Results](results.md) for the returned values and their fields.

## How can I follow progress or stop early?

Pass a `result_callback` to receive each function result as the optimization
runs. Set `report_gradients=True` to receive gradient results too. A callback
can also stop the optimization by returning `True`.

See [Reporting progress](../running/running.md#reporting-progress) and
[Stopping early from the callback](../running/running.md#stopping-early-from-the-callback).

## How can I keep evaluation results?

Pass a handler with the `handlers` argument. `HistoryHandler` keeps every result
it receives; `ResultsHandler` keeps the best or most recent valid result; and
`DataFrameHandler` collects selected fields into pandas or Polars tables.
Handlers can also process events or call custom code.

See [Result Handlers](handlers.md) for built-in handlers, custom handlers, and
sharing handlers across optimizations.

## How can I inspect or export result data?

Result objects include the evaluated variables, objective and constraint
values, gradient information, and optional metadata. To select fields, label
them, or export them as pandas or Polars data, see
[Working with Results](results.md).

## Where do progress messages and optimizer files go?

`ropt` progress messages use Python's standard logging module. The optimizer
backend can also report its own progress with `backend.verbose`.
`optimizer.output_dir` gives the optimizer a directory for its files.
`optimizer.stdout` and `optimizer.stderr` capture the process output streams
while the optimization runs, including output from evaluation code.

See [Logging](../running/logging.md) and
[Choosing an Optimizer](../optimizer_setup/optimizer.md) for details.
