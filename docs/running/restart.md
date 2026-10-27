# Restarting from the Best Point Found

!!! note

    This page starts the new optimization where the previous one ended, under
    the same configuration. What the results at that point still cover when the
    configuration changes between the optimizations is described in
    [Reusing Results at the Starting Point](initial_values.md).

The full script for this example is
[examples/restart.py](https://github.com/TNO-ropt/ropt/blob/main/examples/restart.py).
It restarts the same optimization several times, each time starting from the
best point the previous optimization found.

## Why restart?

A single optimization can stop before truly converging — for example
because it hit its iteration limit while still improving. Restarting
runs [`optimize`][ropt.optimize] again, using the previous result as the
new start point. Since
each call to `optimize` is independent, this is a loop in your own code;
`ropt` needs nothing special to support it.

## Seeing every evaluation

The best point of one optimization is all you need to start the next. To also
see every evaluation across the whole sequence of restarts, pass a
[result callback](running.md#reporting-progress): it is called with each
[`FunctionResults`][ropt.results.FunctionResults] as that evaluation finishes.
A callback belongs to the optimization it is given to, so appending to one list
is what carries the results across the restarts:

```python
reported: list[FunctionResults] = []
```

A **handler**, given with `handlers=`, receives the same results but is an
object with state of its own. One handler can be given to several optimizations,
including optimizations that execute concurrently, and does more than pass
results on:
[`HistoryHandler`][ropt.HistoryHandler] stores every result it receives,
and [`DataFrameHandler`][ropt.DataFrameHandler] collects them into named
DataFrames. [Result Handlers](../results/handlers.md) covers them.

## Restart in a loop

Each iteration runs one optimization, starting from the previous best point,
and appends its results to `reported`:

```python
x0 = INITIAL_VALUES
f0: FunctionResults | None = None
g0: GradientResults | None = None
for _ in range(RESTARTS):
    result = optimize(CONFIG, x0, rosenbrock, result_callback=reported.append, f0=f0, g0=g0)
    assert result.results is not None
    x0 = result.results.variables  # restart from the best point found so far
    f0, g0 = result.results, result.gradient
```

`result.results.variables` is the best point the optimization found — feeding
it back in as `x0` is the entire restart mechanism. After the loop, `reported`
holds every evaluation from every restart, not just the last one:

```python
print(f"evaluations collected across all restarts: {len(reported)}")
print(f"best objective after {RESTARTS} restarts: {result.results.target_objective}")
```

!!! note "Restarting from the last point"
    To restart from the last valid point evaluated rather than the best point,
    create a [`ResultsHandler`][ropt.ResultsHandler] with `what="last"` and
    pass it to `optimize` in `handlers=[handler]`. After the optimization,
    check that `handler.result` is not `None`, then use
    `handler.result.variables` as the next start point. Failed evaluations
    are not retained, so this may not be the last point attempted.

## Not evaluating the restart point again

The loop passes `result.results` as `f0` and `result.gradient` as `g0` to
reuse evaluations already performed at the restart point. See
[Reusing Results at the Starting Point](initial_values.md) for gradient
availability and the rules for reuse when the configuration changes.

## See also

- Restarting concurrent, rather than sequential, optimizations collects into the
  same handler:
  [Result Handlers](../results/handlers.md#sharing-a-handler-across-concurrent-runs).
