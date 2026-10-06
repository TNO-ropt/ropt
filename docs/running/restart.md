# Restarting from the Best Point Found

!!! note

    This page starts the new run where the previous one ended, under the same
    configuration. What the results at that point still cover when the
    configuration changes between the runs is described in
    [Reusing Results at the Starting Point](initial_values.md).

The full script for this example is
[examples/restart.py](https://github.com/TNO-ropt/ropt/blob/main/examples/restart.py).
It restarts the same optimization several times, each time starting from the
best point the previous run found.

## Why restart?

A single optimization run can stop before truly converging — for example
because it hit its iteration limit while still improving. Restarting
runs [`optimize`][ropt.optimize] again, using the previous result as the
new start point. Since
each call to `optimize` is independent, this is a loop in your own code;
`ropt` needs nothing special to support it.

## Seeing every evaluation

The best point of one run is all you need to start the next. To also see every
evaluation across the whole sequence of restarts, pass a
[`report`](running.md#reporting-progress) callback: it is called with each
[`FunctionResults`][ropt.results.FunctionResults] as that evaluation finishes.
A callback belongs to the run it is given to, so appending to one list is what
carries the results across the restarts:

```python
reported: list[FunctionResults] = []
```

A **handler**, given with `handlers=`, receives the same results but is an
object with state of its own. One handler can be given to several runs,
including runs that execute concurrently, and does more than pass results on:
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
    result = optimize(CONFIG, x0, rosenbrock, report=reported.append, f0=f0, g0=g0)
    assert result.results is not None
    x0 = result.results.variables  # restart from the best point found so far
    f0, g0 = result.results, result.gradient
```

`result.results.variables` is the best point the run found — feeding it back in
as `x0` is the entire restart mechanism. After the loop, `reported` holds every
evaluation from every restart, not just the last run's:

```python
print(f"evaluations collected across all restarts: {len(reported)}")
print(f"best objective after {RESTARTS} restarts: {result.results.target_objective}")
```

## Not evaluating the restart point again

The run that found the best point evaluated it, and returns that evaluation on
`result.results`, together with the gradient computed at the same point on
`result.gradient`. Passing the two back as `f0` and `g0` gives the next run
those values, so it does not evaluate the point a second time.

`result.gradient` is `None` when no gradient was computed at the best point, and
`optimize` accepts that: the perturbations at the start point are then evaluated
as usual. Setting
[`evaluation_policy`](../optimizer_setup/gradients.md#evaluation-policy) to
`"speculative"` requests a gradient at every function evaluation, which leaves
a gradient available at whatever point the run returns.

## See also

- Restarting concurrent, rather than sequential, runs collects into the same
  handler:
  [Result Handlers](../results/handlers.md#sharing-a-handler-across-concurrent-runs).
