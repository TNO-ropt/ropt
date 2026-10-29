# Collecting Results

For an overview of the ways to access optimization output, see
[Getting Output](../results/output.md).

## Using a result callback

Pass a function as `result_callback` to receive results as evaluations finish.
[`optimize`][ropt.optimize], [`evaluate`][ropt.evaluate], and
[`evaluate_batch`][ropt.evaluate_batch] accept this parameter. For example,
append each function result to a list:

```python
from ropt import optimize

reported = []
result = optimize(config, x0, objective, result_callback=reported.append)
```

The callback receives one
[`FunctionResults`][ropt.results.FunctionResults] at a time. For
[`optimize`][ropt.optimize], set `report_gradients=True` to also receive
[`GradientResults`][ropt.results.GradientResults]; in that case, the callback
must handle both result types.

The callback does not replace or change the value returned by `optimize`,
which remains the [`OptimizationResult`][ropt.OptimizationResult]. Returning
`True` from the callback asks `optimize` to stop gracefully. Results later in
the same batch are not passed to the callback. The return value is ignored by
`evaluate` and `evaluate_batch`.

Each optimization calls its callback independently. If several optimizations
run concurrently with the same callback, they can call it at the same time. Use
a separate callback for each optimization, make shared state thread-safe, or
use a handler when collecting results across concurrent optimizations.

## Using handlers

An [`optimize`][ropt.optimize] call returns only its single best result. A
**handler** lets you see every result an optimization produces: it is an object
that observes the optimization and processes its results as they arrive.

Pass handlers with the `handlers` argument. Each receives the full result
objects, [`FunctionResults`][ropt.results.FunctionResults] and
[`GradientResults`][ropt.results.GradientResults], rather than the summary
`optimize` returns; see [Working with Results](../results/results.md).

For instance, the [`HistoryHandler`][ropt.HistoryHandler] collects all results:

```python
from ropt import HistoryHandler, optimize

history = HistoryHandler()
result = optimize(config, x0, objective, handlers=[history])
print(len(history.results))   # every evaluation from this optimization
```

Unlike a result callback, a handler can be reused across several
**sequential** calls to `optimize`, accumulating results from all of them.

## Built-in handlers

`ropt` ships a few ready-to-use handlers, all imported from `ropt`:

- **[`HistoryHandler`][ropt.HistoryHandler]** — keeps every result, as used above.
- **[`ResultsHandler`][ropt.ResultsHandler]** — keeps only one result: the best seen so far
  (default), or the most recent.
- **[`DataFrameHandler`][ropt.DataFrameHandler]** — collects results into a `pandas` or `polars` table.

See [Result handlers](../results/handlers.md) for the full
list, and for what a handler shared between concurrent optimizations guarantees.
