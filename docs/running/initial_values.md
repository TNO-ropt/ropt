# Reusing Results at the Starting Point

An optimization begins by evaluating its start point. If those evaluations have
already been performed — by an earlier run, or by one that was interrupted after
them — pass them to [`optimize`][ropt.optimize] as `f0` and `g0` and they
are not performed again.

!!! note

    [Restarting from the Best Point](restart.md) passes these results back under
    an unchanged configuration, where they cover every evaluation at the start
    point. This page is about a configuration that changes between the two runs,
    where part of the start point must still be evaluated.

The full script for this example is
[examples/initial_values.py](https://github.com/TNO-ropt/ropt/blob/main/examples/initial_values.py).
It optimizes an ensemble of two realizations, then restarts from the best point
with five.

## Where the results come from

[`optimize`][ropt.optimize] returns the best evaluation of a run on
`results`, and the gradient computed at that same point on `gradient`. Those are
the two objects a run restarting from that point needs:

```python
first = optimize(config(SCREENING_REALIZATIONS), INITIAL_VALUES, rosenbrock)
```

The configuration sets
[`evaluation_policy`](../optimizer_setup/gradients.md#evaluation-policy) to
`"speculative"`, which computes a gradient at every function evaluation, so a
gradient is available at the point the first run returns.

## Restarting with a larger ensemble

The second run starts at the best point of the first, with five realizations
instead of two, and is given the results recorded there:

```python
optimize(
    config(FULL_REALIZATIONS),
    first.results.variables,
    rosenbrock,
    report=reported.append,
    f0=first.results,
    g0=first.gradient,
)
```

The five-realization ensemble needs 25 evaluations at that point: one per
realization, and one per realization for each of the four perturbations. Ten of
them — the two recorded realizations, unperturbed and at each perturbation —
come from `f0` and `g0`, and the second run evaluates the remaining fifteen. The
first two per-realization objectives are the values the first run returned:

```
per-realization objectives at the restart point: [64.0004475  56.66862051 63.84798703 58.60943629 58.6053663 ]
reused from the first run: [64.0004475  56.66862051]
```

The aggregated objective is recomputed over all five, so it is not the value the
first run reported:

```
objective there, 2 realizations: [60.334534]
objective there, 5 realizations: [60.34637152]
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
starting anywhere else raises `ValueError`. A restart meets this by
construction, since `x0` is the `variables` of the result passed as `f0`.

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
evaluation of each kind; every later point is evaluated normally.

## Several runs at once

[`optimize_many`][ropt.optimize_many] takes either one result shared by
every run, or a sequence with one per run, as it does for `config` and
`metadata`. Sharing one result requires the runs to share `x0`, which is the
case when the same point is optimized under several configurations:

```python
optimize_many([config_a, config_b], x0, function, f0=f0, g0=g0)
```
