# Restarting from the Best Point Found

The full script for this tutorial is
[examples/simple/restart.py](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/restart.py).
It restarts the same optimization several times, each time starting from the
best point the previous run found.

## Why restart?

A single optimization run can stop before truly converging — for example
because it hit its iteration limit while still improving. Restarting simply
runs `optimize` again, using the previous result as the new start point. Since
each call to `optimize` is independent, this is just a loop in your own code;
`ropt` needs nothing special to support it.

## Collecting every result with a handler

`result.variables` from one run is all you need to start the next, but if you
also want to see every evaluation across the whole sequence of restarts — not
just the final result — attach a **handler**. A handler is an object you pass
with `handlers=` that observes every result an optimization produces; unlike
the `report` callback (see [Your First Optimization](optimize.md)), the same
handler can be reused across several calls to `optimize`, accumulating results
as it goes. See [Result handlers](../running/handlers.md#result-handlers) for
the full explanation.

Here we use [`HistoryHandler`][ropt.simple.HistoryHandler], which keeps every
result it sees, in order:

```python
from ropt.simple import HistoryHandler, optimize

history = HistoryHandler()
```

## Restart in a loop

Each iteration runs one optimization, starting from the previous best point,
and feeds its results into `history`:

```python
x0 = INITIAL_VALUES
for _ in range(RESTARTS):
    result = optimize(CONFIG, x0, rosenbrock, handlers=[history])
    x0 = result.variables  # restart from the best point found so far
```

`result.variables` is the best point the run found — feeding it back in as
`x0` is the entire restart mechanism. After the loop, `history.results` holds
every evaluation from every restart, not just the last run's:

```python
print(f"evaluations collected across all restarts: {len(history.results)}")
print(f"best objective after {RESTARTS} restarts: {result.target_objective}")
```

## Next

- The full simple API, including other built-in handlers:
  [Result Handlers](../running/handlers.md#result-handlers).
- Restarting concurrent, rather than sequential, runs needs a **shared**
  handler group instead of a reused one:
  [Running in Parallel](../getting_started/execution.md#collecting-results-from-concurrent-runs).
