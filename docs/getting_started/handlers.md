# Collecting Results with Handlers

So far, `optimize` returned only the single best result. In many cases you want
to see every result to watch progress over the optimization. A **handler** does
this: an object you attach with a `handlers=` argument that observes every
result an optimization produces.

What it collects are the full result objects — `FunctionResults` and
`GradientResults` — rather than the summary `optimize` returns; see
[Working with Results](../running/results.md).

## A handler that collects everything

```python
from ropt.simple import HistoryHandler, optimize

history = HistoryHandler()
result = optimize(config, x0, objective, handlers=[history])
print(len(history.results))   # every evaluation from this run
```

Compare this with the `report` callback from
[Follow the progress](ensemble.md#4-follow-the-progress-optional):
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
    assert result.results is not None
    x0 = result.results.variables   # restart from the best point found so far

print(f"collected {len(history.results)} results across all restarts")
```

Restarting needs nothing special from `ropt`: each call to `optimize` is
independent, so `result.results.variables` — the best point a run found — is
simply the start point for the next one. The runnable script is
[examples/simple/restart.py](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/restart.py),
which [Restarting from the Best Point](../running/restart.md) walks through.

## Other built-in handlers

`ropt` ships a few ready-to-use handlers, all imported from `ropt.simple`:

- **`HistoryHandler`** — keeps every result, as used above.
- **`ResultsHandler`** — keeps only one result: the best seen so far
  (default), or the most recent.
- **`DataFrameHandler`** — collects results into a `pandas` or `polars` table.

See [Result handlers](../running/handlers.md) for the full
list, and how to write your own.
