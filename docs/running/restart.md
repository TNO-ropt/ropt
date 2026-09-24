# Restarting from the Best Point Found

The full script for this example is
[examples/simple/restart.py](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/restart.py).
It restarts the same optimization several times, each time starting from the
best point the previous run found.

## Why restart?

A single optimization run can stop before truly converging — for example
because it hit its iteration limit while still improving. Restarting
runs [`optimize`][ropt.simple.optimize] again, using the previous result as the
new start point. Since
each call to `optimize` is independent, this is a loop in your own code;
`ropt` needs nothing special to support it.

## Collecting every result with a handler

The best point of one run is all you need to start the next, but if you
also want to see every evaluation across the whole sequence of restarts — not
just the final result — attach a **handler**. A handler is an object you pass
with `handlers=` that observes every result an optimization produces; unlike
the `report` callback (see
[Running Optimizations](running.md#reporting-progress)), the same
handler can be reused across several sequential calls to `optimize`, accumulating
results as it goes. See [Result handlers](handlers.md) for
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
    assert result.results is not None
    x0 = result.results.variables  # restart from the best point found so far
```

`result.results.variables` is the best point the run found — feeding it back in
as `x0` is the entire restart mechanism. After the loop, `history.results` holds
every evaluation from every restart, not just the last run's:

```python
print(f"evaluations collected across all restarts: {len(history.results)}")
print(f"best objective after {RESTARTS} restarts: {result.results.target_objective}")
```

## See also

- Restarting concurrent, rather than sequential, runs collects into the same
  handler:
  [Result Handlers](handlers.md#sharing-a-handler-across-concurrent-runs).
