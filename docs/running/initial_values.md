# Reusing Results at the Starting Point

An optimization begins by evaluating its start point. If those evaluations have
already been performed — by an earlier run, or by one that was interrupted after
them — pass them to [`optimize`][ropt.simple.optimize] as `f0` and `g0` and they
are not performed again.

!!! note

    This is not [Restarting from the Best Point](restart.md), where a new run
    begins where a previous one ended. Here the new run begins at the *same*
    point as the recorded one, and only the evaluations at that point are
    avoided.

The full script for this example is
[examples/simple/initial_values.py](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/initial_values.py).

## Recording the results

The results of the first run are collected with a
[`HistoryHandler`][ropt.simple.HistoryHandler], from which the first
[`FunctionResults`][ropt.results.FunctionResults] and the first
[`GradientResults`][ropt.results.GradientResults] are the ones at the start
point:

```python
history = HistoryHandler()
optimize(CONFIG, INITIAL_VALUES, first, handlers=[history])
f0 = next(item for item in history.results if isinstance(item, FunctionResults))
g0 = next(item for item in history.results if isinstance(item, GradientResults))
```

How these are kept between the runs is up to you: held in memory, pickled to
disk, or stored in a database.

## Supplying them to the next run

The second run starts from the same vector and is given both results. Its
realization weights differ from the first run's:

```python
config = deepcopy(CONFIG)
config["realizations"]["weights"] = [3.0, 1.0, 1.0]
optimize(config, INITIAL_VALUES, second, handlers=[reused], f0=f0, g0=g0)
```

The second run evaluates nothing at the start point. Its objective there differs
from the first run's, because the recorded per-realization values are aggregated
under the new weights:

```
objective at the starting point, first run:  [1.1]
objective at the starting point, second run: [0.96]
```

Either argument may be given on its own. With only `f0`, the perturbations at
the start point are evaluated as usual; with only `g0`, the unperturbed
evaluations are.

## What is read from the results

| From `f0` | From `g0` |
| --- | --- |
| `evaluations.objectives` | `evaluations.perturbed_objectives` |
| `evaluations.constraints` | `evaluations.perturbed_constraints` |
| `realizations.evaluated_realizations` | `realizations.evaluated_realizations` |
| `variables` | `variables` and `perturbed_variables` |

These are the per-realization values as the evaluation function returned them,
before any scaling, together with the record of which realizations were
evaluated. Everything else on the two objects is ignored, including the
aggregated functions and gradients, the realization weights, the metadata and
the batch id. All of those are recomputed by the second run under its own
configuration.

## What may differ between the two runs

The number of realizations and the number of perturbations may both grow or
shrink. Realization weights, realization filters, objective weights, scales,
offsets, `maximize` flags, function estimators, `merge_realizations` and the
sampler settings may all change.

What may not change is the meaning of the recorded columns: the number of
objectives, the number of nonlinear constraints, whether constraints are present
at all, and the number of variables. A mismatch raises `ValueError` before
anything is evaluated.

The start point is checked as well. The `variables` on `f0` and on `g0` must
match `x0`, which is also what makes the two results belong together. A run
starting anywhere else raises `ValueError`.

## Realizations and perturbations that were not recorded

`evaluated_realizations` says which realizations the recorded run computed. One
that it skipped — because its weight was zero, or because it did not exist yet —
is evaluated by the new run, and the rest are taken from the record. The same
holds per perturbation: a run configured with more perturbations than were
recorded evaluates the additional ones and reuses the rest.

Perturbed points that were recorded are used as they were; the new run draws
points only for the perturbations it adds.

## Things to know

**Realizations correspond by position.** Realization 3 of the recorded run is
realization 3 of the new one. If the realizations were reordered between the
runs, the values are attached to the wrong ones, and nothing detects this.

**A failed realization counts as evaluated.** A realization that returned `NaN`
is part of the record and is not evaluated again. Supply a record without it if
it should be retried.

**Only the start point is covered.** `f0` and `g0` are used once, by the first
evaluation of each kind; every later point is evaluated normally. Under an
unchanged configuration the recorded values are the ones the run would have
computed, so it follows the same trajectory as the recorded run.

## Several runs at once

[`optimize_many`][ropt.simple.optimize_many] takes either one result shared by
every run, or a sequence with one per run, as it does for `config` and
`metadata`. Sharing one result requires the runs to share `x0`, which is the
case when the same point is optimized under several configurations:

```python
optimize_many([config_a, config_b], x0, function, f0=f0, g0=g0)
```
