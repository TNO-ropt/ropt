# Collecting Results with Handlers

So far, `optimize` returned only the single best result. Sometimes you want to
see every result instead — to watch progress across several runs, or to log
every evaluation. A **handler** does this: an object you attach with
`handlers=` that observes every result an optimization produces.

## A handler that collects everything

```python
from ropt.simple import HistoryHandler, optimize

history = HistoryHandler()
result = optimize(config, x0, objective, handlers=[history])
print(len(history.results))   # every evaluation from this run
```

Compare this with the `report` callback from
[Deterministic Optimization](deterministic.md#3-follow-the-progress-optional):
`report` is called once per evaluation, for one run. A handler is more
general — it keeps or reacts to results, and, unlike `report`, the same
handler can be reused across several **sequential** calls to `optimize`,
accumulating results from all of them.

## Example: restarting from the best point

For instance, restart the same optimization from the best point the previous
run found, while collecting every result from every restart in one handler:

```python
x0 = initial_values
for _ in range(3):
    result = optimize(config, x0, objective, handlers=[history])
    x0 = result.variables   # restart from the best point found so far

print(f"collected {len(history.results)} results across all restarts")
```

Restarting needs nothing special from `ropt`: each call to `optimize` is
independent, so `result.variables` — the best point a run found — is simply
the start point for the next one. See
[Restarting from the Best Point](../tutorials/restart.md) for the full,
runnable version of this example.

## Other built-in handlers

`ropt` ships a few ready-to-use handlers, all imported from `ropt.simple`:

- **`HistoryHandler`** — keeps every result, as used above.
- **`ResultsHandler`** — keeps only one result: the best seen so far
  (default), or the most recent.
- **`DataFrameHandler`** — collects results into a `pandas` or `polars` table.

See [Result handlers](../running/handlers.md#result-handlers) for the full
list, and how to write your own.

## Where to next

- Collecting results from runs that overlap in time, instead of one after
  another: [Running in Parallel](execution.md#collecting-results-from-concurrent-runs).
- The complete simple API: [Result Handlers](../running/handlers.md#result-handlers).
- A full worked example: [Restarting from the Best Point](../tutorials/restart.md).
