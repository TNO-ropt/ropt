# Choosing an Optimizer

Every run needs an algorithm and a criterion for stopping. `ropt` supplies a
default for both — SciPy's SLSQP, running until it converges — so neither has to
be configured for a first run. This page covers what to change when the default
is not appropriate.

The two settings live in different places: **`backend` selects and configures
the algorithm**, while **`optimizer` sets the limits `ropt` enforces around
it**. The full field reference is under
[`backend`](configuration_sections.md#backend) and
[`optimizer`](configuration_sections.md#optimizer).

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

    Not every method supports every kind of constraint, and a mismatch is
    rejected when the run starts rather than ignored; see
    [`backend`](configuration_sections.md#backend) for what that error is. The
    one case that *is* ignored silently is integer variables; see
    [Discrete and Mixed-Integer Variables](discrete.md).

## Running the method in a separate process { #external-backend }

Prefix the method with `external/` to run the optimization algorithm in a
process of its own:

```python
"backend": {"method": "external/scipy/slsqp"}
```

`ropt` spawns a child process, creates the named backend there, and lets it
drive the optimization. The function and gradient evaluations still happen in
the original process: the child sends each set of variables back, the parent
evaluates it as usual, and the values are passed to the child. An error raised
in the child is re-raised in the parent. This is unrelated to evaluating in
parallel, which is about where your *objective* runs; see
[Evaluating in Parallel](../running/parallel.md).

This is useful when a backend cannot safely share a process with the rest of
your program — for example one that crashes the interpreter, leaks memory,
keeps state between runs, or links against native libraries that clash with
your other dependencies.

It is also the answer for a backend that **cannot run concurrently in-process**.
Some optimizers need a working directory of their own, write to a file whose
name is fixed, or keep state inside the library that a second simultaneous run
corrupts. What such a backend rules out is not merely a second run of its own
kind: changing the working directory applies to the whole process, so it breaks
another run's relative output path, and any file your evaluation function opens
by relative name, just as surely. Each backend states in its own documentation
whether this applies to it; where it does, `external/` is what lets it run
alongside anything else, because the state it needs is then its own. This
matters as soon as runs overlap — see
[Many Runs at Once](../running/many_runs.md).

Two details differ from the other backends:

- The method must name the delegate in full, as `external/plugin/method` or
  `external/method`. The `external/` prefix is removed and the rest is resolved
  like any other method string. `external` is never selected implicitly, so it
  is used only when you ask for it by name.
- The problem is sent to the child process, so everything describing it must be
  serializable. The built-in plugins are, and so is any plugin class defined in
  a module that can be imported. Only if you pass a plugin instance of a class
  defined inside a function or a notebook do you need the optional
  `cloudpickle` extra (see
  [Installation](../getting_started/installation.md#optional-extras)). Without
  it the two differ in *where* they fail: a class defined inside a function
  cannot be sent at all, and is refused here with an
  [`ExecutionError`][ropt.exceptions.ExecutionError]; a class defined in a
  notebook is sent by name, and the failure arrives from the child, which
  reports the name it could not find. Your objective function is never
  affected: it stays in this process.

## Gradient-based or gradient-free

Most `ropt` methods are gradient-based, and since a black-box objective has no
analytic derivative, they rely on the estimated gradients described in
[Stochastic Gradients](gradients.md). That estimate costs extra evaluations at
perturbed points on every iteration.

A gradient-free method such as `differential_evolution` avoids that entirely: it
evaluates many candidate points instead of following a gradient. It needs bounds
on every variable, ignores `perturbation_magnitudes`, and is the only choice
when any variable is an integer.

## Limiting the length of a run

Four settings can stop a run, and they are **split across the two sections**:

| Setting | Section | Limits |
| --- | --- | --- |
| `max_batches` | `optimizer` | calls made to your evaluation function |
| `max_functions` | `optimizer` | individual objective evaluations, across all batches |
| `max_iterations` | `backend` | iterations of the algorithm itself |
| `convergence_tolerance` | `backend` | the improvement below which the algorithm stops |

The division follows who enforces the limit. `ropt` counts batches and function
evaluations itself, so those belong to `optimizer`. Iterations and convergence
are the algorithm's own criteria — their exact meaning depends on the method,
and not every backend supports them — so they belong to `backend`.

```python
CONFIG = {
    "variables": {"variable_count": 5, "perturbation_magnitudes": 1e-6},
    "optimizer": {"max_functions": 500},
    "backend": {"method": "scipy/slsqp", "convergence_tolerance": 1e-8},
}
```

Which limit applies depends on which resource constrains the run. When each
evaluation is expensive, `max_functions` bounds the total number of evaluations
directly. When the evaluations within a batch are submitted concurrently,
`max_batches` bounds the number of sequential submissions, which determines the
elapsed time. A batch may contain many evaluations, so the two are not
interchangeable.

`convergence_tolerance` is compared against the quantities the optimizer works
with, which are **scaled**. An objective that has been divided by a large number
reaches any absolute tolerance sooner; see [Objectives](objectives.md) for how
scaling changes what the optimizer sees.

To stop on a condition of your own rather than a count, return `True` from the
`report` callback; that is covered in
[Running Optimizations](../running/running.md), with
[examples/simple/stopping.py](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/stopping.py)
as the runnable version.

## Output from the optimizer

Backends can report their own progress, which is disabled by default:

```python
"backend": {"verbose": True}
```

`verbose` determines whether output is produced at all. Where it goes is separate:
`optimizer.stdout` and `optimizer.stderr` capture it to files, resolved against
`optimizer.output_dir`. Capture rewires process-global state, so only one run at
a time can use it — leave those unset on runs that overlap. See
[`optimizer`](configuration_sections.md#optimizer) for the details.

## See also

- Watching results as they arrive, instead of reading optimizer output:
  [Result Handlers](../results/handlers.md).
- When a run stops earlier than expected, or returns no result at all:
  [Troubleshooting](../troubleshooting/index.md).
