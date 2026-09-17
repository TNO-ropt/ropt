# Choosing an Optimizer

Every run needs an algorithm and a budget. `ropt` picks a reasonable default for
both — SciPy's SLSQP, running until it converges — so neither has to be
configured to get started. This page covers what to change when the default is
not what you want.

The two settings live in different places, which is worth knowing up front:
**`backend` selects and configures the algorithm**, while **`optimizer` governs
the run around it**. The full field reference is under
[`backend`](configuration.md#backend) and
[`optimizer`](configuration.md#optimizer).

## Selecting a method

A method is named by a `"plugin/method"` string:

```python
"backend": {"method": "scipy/slsqp"}
```

The default is `"scipy/default"`, which is SLSQP. The plugin part is the name a
backend registers itself under, and the method part is any name that backend
declares; both are case-insensitive, so `"SciPy/SLSQP"` and `"scipy/slsqp"` are
the same. The plugin part may be omitted when only one installed plugin provides
the method. See [Method strings](configuration.md#method-strings) for the full
rules, and [Optimizer Backends](../reference/backend.md) for the methods each
backend supports.

Options that belong to the algorithm itself, rather than to `ropt`, go in
`options`:

```python
"backend": {"method": "scipy/differential_evolution", "options": {"rng": 4}}
```

!!! note

    Not every method supports every kind of constraint. `ropt` raises an error
    at the start of a run when the configured method cannot handle what the
    configuration declares, rather than silently ignoring it. The one case where
    something *is* ignored silently is integer variables; see
    [Discrete and Mixed-Integer Variables](discrete.md).

## Gradient-based or gradient-free

Most `ropt` methods are gradient-based, and since a black-box objective has no
analytic derivative, they rely on the estimated gradients described in
[Stochastic Gradients](gradients.md). That estimate costs extra evaluations at
perturbed points on every iteration.

A gradient-free method such as `differential_evolution` skips that entirely: it
evaluates many candidate points instead of following a slope. It needs bounds on
every variable, ignores `perturbation_magnitudes`, and is the only choice when
any variable is an integer.

## Budgeting the run

Four settings can stop a run, and they are **split across the two sections**:

| Setting | Section | Limits |
| --- | --- | --- |
| `max_batches` | `optimizer` | calls made to your evaluation function |
| `max_functions` | `optimizer` | individual objective evaluations, across all batches |
| `max_iterations` | `backend` | iterations of the algorithm itself |
| `convergence_tolerance` | `backend` | how small an improvement still counts |

The division follows who enforces the limit. `ropt` counts batches and function
evaluations itself, so those belong to `optimizer`. Iterations and convergence
are the algorithm's own notions — their exact meaning depends on the method, and
not every backend supports them — so they belong to `backend`.

```python
CONFIG = {
    "variables": {"variable_count": 5, "perturbation_magnitudes": 1e-6},
    "optimizer": {"max_functions": 500},
    "backend": {"method": "scipy/slsqp", "convergence_tolerance": 1e-8},
}
```

Which to reach for depends on what is scarce. When each evaluation is expensive,
`max_functions` caps the total cost directly. When evaluations are submitted in
parallel, `max_batches` caps the number of sequential submission rounds, which
is usually the thing that determines wall-clock time. A batch may contain many
evaluations, so the two are not interchangeable.

`convergence_tolerance` is compared against the quantities the optimizer works
with, which are **scaled**. An objective that has been divided by a large number
reaches any absolute tolerance sooner; see [Objectives](objectives.md) for how
scaling changes what the optimizer sees.

To stop on a condition of your own rather than a count, return `False` from the
`report` callback; that is covered in
[Running Optimizations](../running/running.md), with
[examples/simple/stopping.py](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/stopping.py)
as the runnable version.

## Seeing what the optimizer is doing

Backends can report their own progress, which is off by default:

```python
"backend": {"verbose": True}
```

That decides whether there is any output. Where it goes is separate:
`optimizer.stdout` and `optimizer.stderr` capture it to files, resolved against
`optimizer.output_dir`. Capture rewires process-global state, so only one run at
a time can use it — leave those unset on runs that overlap. See
[`optimizer`](configuration.md#optimizer) for the details.

## See also

- Watching results as they arrive, instead of reading optimizer output:
  [Result Handlers](../running/handlers.md).
- When a run stops earlier than expected, or returns no result at all:
  [Common Pitfalls](../troubleshooting/index.md).
