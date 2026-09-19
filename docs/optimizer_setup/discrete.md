# Discrete and Mixed-Integer Variables

By default every variable is continuous. Marking some or all of them as
integer-valued takes one field, `variables.types`, but it changes what the rest
of the configuration has to look like: an integer variable cannot be
differentiated, so the problem needs a method that searches without gradients,
and that method needs bounds.

There are two runnable scripts for this page:
[examples/simple/discrete.py](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/discrete.py),
where every variable is an integer, and
[examples/simple/mixed.py](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/mixed.py),
where continuous and integer variables appear in one problem.

!!! warning

    Only `differential_evolution` handles integer variables. **The other SciPy
    methods silently treat them as continuous** — no error, only a fractional
    answer to a problem you meant to be discrete. Choosing the method is not
    optional here.

## All variables integer

`discrete.py` maximizes `min(3x, y)` over two integers, subject to
`x + y <= 10`. The `types` field takes a single value that applies to every
variable:

```python
--8<-- "examples/simple/discrete.py:config"
```

Three things go together. `types` marks the variables as integers; the `backend`
section selects `differential_evolution`, the only method that will respect
that; and `lower_bounds` / `upper_bounds` are mandatory, because that method
searches within a box rather than stepping from a start point.

[`VariableType`][ropt.enums.VariableType] comes from `ropt.enums`, not from
`ropt.simple`:

```python
from ropt.enums import VariableType
```

The `types` field, and every other field of the variables section, is described
under [`variables`](configuration_sections.md#variables).

The objective is an ordinary evaluation function. It receives the variables as
floats that happen to hold integral values:

```python
--8<-- "examples/simple/discrete.py:objective"
```

The script imposes `x + y <= 10` as a nonlinear constraint by default, and as a
linear one with `--linear`. Both forms work unchanged with integer variables;
see [Constraints](constraints.md).

## Mixing continuous and integer variables

To make only some variables discrete, give `types` one entry per variable
instead of a single value. `mixed.py` does this for the first four variables of
an ensemble Rosenbrock problem, keeping two continuous and two integer:

```python
--8<-- "examples/simple/mixed.py:config"
```

Nothing else changes. The realizations, the objective and the call to
[`optimize`][ropt.simple.optimize]
are the same as in [Ensemble-Based Optimization](../getting_started/ensemble.md)
— being partly discrete is a property of the variables, not of the problem
around them.

## What to expect from the search

A gradient-free method reaches an answer by evaluating many points rather than
by following a gradient, and the cost profile differs accordingly:

- `perturbation_magnitudes` is unused. No perturbations are evaluated, because
  no gradient is estimated.
- Limit the run with [`max_functions`](configuration_sections.md#optimizer) or
  `max_iterations` rather than a convergence tolerance.
- The result is reproducible only if the method's own generator is seeded —
  hence `"options": {"rng": 4}` in both scripts.
