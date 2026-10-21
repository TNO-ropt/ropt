# Collecting Results with Handlers

So far, [`optimize`][ropt.optimize] returned only the single best result. In
many cases you want to see every result to watch progress over the optimization
using a **handler**: an object that observes every result an optimization
produces.

Multiple handlers can be passed via the `handlers` argument, each receving the
full result objects, [`FunctionResults`][ropt.results.FunctionResults] and
[`GradientResults`][ropt.results.GradientResults]. rather than the summary
`optimize` returns; see [Working with Results](../results/results.md).

For instance, the [`HistoryHandler`][ropt.HistoryHandler] collects all results:

```python
from ropt import HistoryHandler, optimize

history = HistoryHandler()
result = optimize(config, x0, objective, handlers=[history])
print(len(history.results))   # every evaluation from this optimization
```

Compare this with the result callback from
[Follow the progress](ensemble.md#4-follow-the-progress-optional):
The result callback is called once per evaluation, for one optimization. A handler is more
general — it keeps or reacts to results, and, unlike the result callback, the same
handler can be reused across several **sequential** calls to `optimize`,
accumulating results from all of them.

## Built-in handlers

`ropt` ships a few ready-to-use handlers, all imported from `ropt`:

- **[`HistoryHandler`][ropt.HistoryHandler]** — keeps every result, as used above.
- **[`ResultsHandler`][ropt.ResultsHandler]** — keeps only one result: the best seen so far
  (default), or the most recent.
- **[`DataFrameHandler`][ropt.DataFrameHandler]** — collects results into a `pandas` or `polars` table.

See [Result handlers](../results/handlers.md) for the full
list, and for what a handler shared between concurrent optimizations guarantees.
