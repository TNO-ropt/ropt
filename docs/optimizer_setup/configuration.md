# Configuration

Every `ropt` optimization run is described by a configuration dictionary that is
validated into an [`EnOptContext`][ropt.context.EnOptContext] object. This page
walks through the top-level keys of that dictionary, the rules that apply to all
of them, and how to compose them for typical problems.

For the full schema, see the reference page for
[`EnOptContext`][ropt.context.EnOptContext] and
[Configuration Classes](../reference/config.md).

## Top-level layout

```python
CONFIG = {
    "variables": {...},                        # required
    "objectives": {...},                       # optional
    "linear_constraints": {...},               # optional
    "nonlinear_constraints": {...},            # optional
    "realizations": {...},                     # optional
    "optimizer": {...},                        # optional
    "backend": {...},                          # optional
    "gradient": {...},                         # optional
    "realization_filters": [...],              # optional, tuple
    "function_estimators": [...],              # optional, tuple
    "samplers": [...],                         # optional, tuple
    "names": {...},                            # optional, for labelled output
}
```

Only `variables` is required. Each value is either a plain dict (which Pydantic
validates against the corresponding config class) or a list/tuple of such
dicts for the plugin-bearing fields.

## Rules that apply everywhere

### Pydantic validation

All configuration dictionaries are validated using
[Pydantic](https://docs.pydantic.dev/). This means inputs are automatically
coerced to the expected types when possible. For example, you can pass a `list`
wherever a `tuple` is expected, or a plain `list` of numbers wherever a NumPy
array is required — Pydantic will handle the conversion during validation.

Some values are also adjusted during validation. For instance, the `weights`
fields in `objectives` and `realizations` are normalized to sum to 1:

```python
"realizations": {"weights": [1.0, 1.0, 1.0]}  # stored as [0.333, 0.333, 0.333]
```

### Broadcasting

Many per-variable, per-objective, or per-constraint fields are NumPy arrays. A
size-1 value is broadcast to match the relevant count, for example:

```python
"variables": {
    "variable_count": 5,
    "lower_bounds": 0.0,        # broadcast to [0, 0, 0, 0, 0]
    "upper_bounds": [1, 2, 3, 4, 5],
}
```

Length-mismatched arrays raise a validation error.

### Sharing optimizer components by key

Each tuple-typed field holds the optimizer **components** of the corresponding
kind:

| Field                              | Component type                                                                  |
| ---------------------------------- | ------------------------------------------------------------------------------- |
| `realization_filters`              | [`RealizationFilter`][ropt.realization_filter.RealizationFilter]                |
| `function_estimators`              | [`FunctionEstimator`][ropt.function_estimator.FunctionEstimator]                |
| `samplers`                         | [`Sampler`][ropt.sampler.Sampler]                                               |

You usually specify each component with a small config dict; see
[Providing optimizer components](#providing-optimizer-components) below.

Other config sections refer to these components **by key**. Give them as a
mapping to choose the keys yourself, or as a list, which is keyed by position —
so a list entry is reached by the integer that used to index it. For example,
[`VariablesConfig`][ropt.config.VariablesConfig] has a `samplers` field that
selects a sampler for each variable:

```python
"samplers": [
    {"method": "scipy/default"},   # index 0
    {"method": "scipy/sobol"},     # index 1
],
"variables": {
    "variable_count": 4,
    "samplers": [0, 0, 1, 1],   # variables 0,1 use sampler 0; 2,3 use sampler 1
},
```

The same thing written with names:

```python
{
    "samplers": {"coarse": {...}, "fine": {...}},
    "variables": {
        "samplers": ["coarse", "coarse", "fine", "fine"],
    },
}
```

Use a single key to share one component across all elements; thanks to
broadcasting, a single value (the default `"0"`) is sufficient.

For optional fields like `realization_filters`, `None` (the default) leaves the
corresponding element unfiltered. Any other key must exist, or building the
context fails with an error naming the unknown key.

### Providing optimizer components

Each entry ends up as an object that implements the component's base class (from
the table above). You can build that object yourself, but usually you just
provide a config dict and let the plugin system build it. Each tuple element
accepts any of three equivalent forms, and a Pydantic validator converts it to
the required object:

1. **A plain `dict`** — *the usual case*. Give a small dictionary with a `method`
   field (and optional `options`), and `ropt` builds the object for you through
   the plugin system:

    ```python
    "samplers": [
        {"method": "scipy/default"},   # index 0  -> SciPySampler
        {"method": "scipy/sobol"},     # index 1  -> SciPySampler
    ],
    ```

    The validator looks up a plugin in the `ropt.plugins` sub-package by the
    `method` field (`"plugin/method"` form, or just `"method"` for implicit
    discovery) and builds the object, applying any `options`.

2. **A typed config object**, for example
   [`SamplerConfig`][ropt.config.SamplerConfig]. This is the same as the dict
   form — the dict is validated into exactly this object — but lets you build it
   explicitly in Python.

3. **An already-constructed object** — *advanced*. Pass an instance of a built-in
   class (for example a `SciPySampler` from `ropt.sampler.scipy`) or of your own
   `Sampler` subclass, and it is used as-is. This is mainly for when you write
   the Python object yourself and want to provide it directly, without
   registering it as a plugin.

The same pattern applies to `backend`, `function_estimators`, and
`realization_filters`. You can mix these forms
freely — for example, a hand-built `Sampler` instance alongside a dict-configured
one in the same tuple. The dict and config-object forms resolve the `method`
through the plugin system, so the plugin must be registered; providing an object
directly does not.

### Method strings

All `method` fields use the same naming convention:

- **`"plugin/method"`** — *explicit*: use method `method` from the plugin named
  `plugin`. For example, `"scipy/default"` selects the `default` method from
  the `scipy` plugin.
- **`"method"`** — *implicit*: omit the plugin name and let `ropt` search all
  registered plugins for one that supports `method`. This is convenient when
  only one plugin provides the method, but ambiguous if multiple plugins expose
  the same name.

The plugin part corresponds to the name under which the plugin is registered
(via an entry point); the method part is any string that the plugin's
`is_supported()` classmethod accepts. For example, the built-in SciPy backend
plugin is named `scipy` and supports methods like `"default"`, `"SLSQP"`, and
`"L-BFGS-B"`.

Both the plugin name and the method name are case-insensitive, so
`"SciPy/SLSQP"`, `"scipy/slsqp"`, and `"SCIPY/Slsqp"` all resolve to the
same backend.

The `backend` field accepts one further form, `"external/..."`, which runs the
named backend in a separate process; see [Running the optimizer in a separate
process](#external-backend).

### Immutability

The configuration objects an [`EnOptContext`][ropt.context.EnOptContext] holds
are frozen, so an individual setting cannot be changed in place. The context
itself is not: replacing one of its fields wholesale is not prevented, but
nothing re-runs the work construction did — bounds, perturbation magnitudes and
linear constraints are all scaled at that point — so
the result is inconsistent. To change settings, build a new context from a
modified dict.

!!! warning

    Treat an `EnOptContext` as read-only after construction. Do not try to
    serialize and round-trip them (for example, to/from JSON). Some parameters
    are scaled during construction in a way that cannot be undone, so
    building an `EnOptContext` from those serialized values would scale
    them again, incorrectly. NumPy arrays and plugin instances may also not
    come back unchanged from a round-trip. Persist the raw input dicts instead
    if you intend to
    modify the values.

## Section reference

### `variables` — [`VariablesConfig`][ropt.config.VariablesConfig] { #variables }

Defines the decision variables for the optimization problem.

The `variable_count` field is required and determines the total number of
variables, including both free and fixed variables.

The `lower_bounds` and `upper_bounds` fields define the bounds for each
variable. They are broadcasted to match the number of variables and default to
$-\infty$ and $+\infty$, respectively. `numpy.nan` values in these arrays
indicate unbounded variables and are converted to `numpy.inf` with the
appropriate sign.

The optional `types` field allows assigning a
[`VariableType`][ropt.enums.VariableType] to each variable (continuous or
integer). If not provided, all variables default to continuous
([`VariableType.REAL`][ropt.enums.VariableType.REAL]). Integer variables are
only honored by methods that support them; in the SciPy backend that is
`differential_evolution` (see
[`SciPyBackend`][ropt.backend.scipy.SciPyBackend]).

The optional `mask` field is a boolean array that indicates which variables are
free to change during optimization (default: all `True`, meaning all variables
are free). `True` means the variable is free; `False` means it is fixed.

```python
"variables": {
    "variable_count": 3,
    "lower_bounds": -1,
    "upper_bounds": [1, 2, 3],
    "types": "real",              # default; or "integer" for discrete variables
    "mask": [True, True, False],  # third variable is fixed
    "perturbation_magnitudes": 1e-5,
}
```

#### Variable perturbations { #variable-perturbations }

The `variables` section also stores information needed to generate perturbed
variables for stochastic gradient estimation (see [Stochastic
Gradients](gradients.md)).

Perturbations are generated by [`Sampler`][ropt.sampler.Sampler] instances
configured in the
[`samplers`](#function_estimators-realization_filters-samplers)
tuple. The `samplers` field of `variables` assigns each variable to a sampler by
its index into that tuple (default: `0`, meaning the first sampler). Unless
explicitly configured otherwise, the default sampler method is
`"scipy/default"`, which draws perturbations from a standard normal distribution
$N(0, 1)$.

The generated perturbation values are scaled by `perturbation_magnitudes`
(default: `0.005`) and can be modified based on `perturbation_types` (see
[`PerturbationType`][ropt.enums.PerturbationType]):

- [`ABSOLUTE`][ropt.enums.PerturbationType.ABSOLUTE] (default): the
  perturbation magnitude is added directly to the variable value.
- [`RELATIVE`][ropt.enums.PerturbationType.RELATIVE]: the magnitude is scaled
  based on the variable's bounds.

Perturbed variables may violate the defined bounds. The `boundary_types` field
specifies how to handle such violations (see
[`BoundaryType`][ropt.enums.BoundaryType]). The default,
[`MIRROR_BOTH`][ropt.enums.BoundaryType.MIRROR_BOTH], mirrors perturbations
back into the valid range.

The `seed` value (default: `1`) ensures consistent results across repeated runs.
To obtain unique results for each optimization run, modify the seed. A common
approach is to use a tuple with a unique ID as the first
element, ensuring reproducibility across nested and parallel evaluations.

!!! tip "Named constants"

    The defaults above are defined as named constants in
    [`ropt.config.constants`][ropt.config.constants]:
    [`DEFAULT_SEED`][ropt.config.constants.DEFAULT_SEED],
    [`DEFAULT_PERTURBATION_MAGNITUDE`][ropt.config.constants.DEFAULT_PERTURBATION_MAGNITUDE],
    [`DEFAULT_PERTURBATION_TYPE`][ropt.config.constants.DEFAULT_PERTURBATION_TYPE], and
    [`DEFAULT_PERTURBATION_BOUNDARY_TYPE`][ropt.config.constants.DEFAULT_PERTURBATION_BOUNDARY_TYPE].

#### Scaling the variables { #variable-scales }

Variables reach the optimizer as $y = (x - o)/s$, using the `scales` and
`offsets` fields, and are reported back to you as $x = s\,y + o$. Both
directions come from the same two arrays, so they cannot disagree.

```python
"variables": {
    "variable_count": 2,
    "lower_bounds": [0.0, 100.0],
    "upper_bounds": [1.0, 600.0],
    "scales": [1.0, 500.0],
    "offsets": [0.0, 100.0],
}
```

Scaling matters when variables differ by orders of magnitude. An optimizer takes
a step of the same size in every direction, and judges convergence with one
tolerance for all of them; both of those are only meaningful if the variables
are comparable in size. The example above puts both variables in the range
$[0, 1]$.

Deriving the scales and offsets from the bounds like that is the common case,
and [`scales_and_offsets_from_bounds`][ropt.utils.scales_and_offsets_from_bounds]
does it for you:

```python
from ropt.utils import scales_and_offsets_from_bounds

scales, offsets = scales_and_offsets_from_bounds([0.0, 100.0], [1.0, 600.0])
```

Scales must be positive: a scale is a change of units, and nothing else. The
default is a scale of 1 and an offset of 0, which is the identity. There is no
`auto_scale` for variables, because there is nothing to estimate one from: the
bounds are the only information available before the run starts, and using them
is a choice, not a default.

Everything that describes the variables moves with them. The bounds are mapped,
and so are the [perturbation magnitudes](#variable-perturbations) of type
`ABSOLUTE`; a magnitude is a distance rather than a position, so only the scale
applies to it, not the offset. `RELATIVE` magnitudes are a fraction of the bound
range and are already dimensionless, so they are left alone. Linear constraints
follow the variables too, as described [below](#linear-constraint-scales).

The scales and offsets apply to every variable, including the ones fixed by
`mask`. This keeps the scaling uniform, and a fixed variable maps back
to exactly the value you gave it.

What you see in the results follows from that:

- Variables and perturbed variables are reported unscaled.
- Distances to the variable bounds, in
  [`ConstraintInfo`][ropt.results.ConstraintInfo], are multiplied by the scale.
  An offset is a shift shared by a value and its bound, so it cancels out of the
  distance between them.

### `objectives` — [`ObjectiveFunctionsConfig`][ropt.config.ObjectiveFunctionsConfig] { #objectives }

`ropt` supports multi-objective optimization. Multiple objectives are combined
into a single value by summing them after weighting. The `weights` field
determines the weight of each objective function, and its length defines the
number of objectives (default: `[1.0]`, meaning a single objective). The weights
are automatically normalized to sum to 1 (for example, `[1, 1]` becomes `[0.5, 0.5]`).

```python
"objectives": {"weights": [0.6, 0.4]}
```

Weights must not be negative, and must not all be zero. A zero weight is
allowed, and disables its objective.

Objective functions can optionally be processed using
[realization filters](realization_filters.md) and
[function estimators](function_estimators.md). Both fields select an object by
its key in the corresponding mapping defined in
[`EnOptContext`][ropt.context.EnOptContext].

- `realization_filters`: default `None` (no filter applied).
- `function_estimators`: default `"0"` (the first function estimator). Unless
  explicitly configured otherwise, the default function estimator method is
  `"default/default"`, which computes a weighted average of the per-realization
  values.

An out-of-range `realization_filters` index means no filter is applied to that
objective. An out-of-range `function_estimators` index leaves that objective
unestimated, and its value is reported as `NaN`.

#### Scaling objectives { #objective-scales }

Objectives are passed to the optimizer divided by the `scales` field, and
reported back to you multiplied by it again. This changes the numbers the
optimizer sees, which matters when objectives differ by orders of magnitude or
when a method has absolute tolerances.

With a single objective that is all it changes: dividing by a positive constant
leaves the optimum where it was. With several objectives it does more. The
optimizer minimizes $\sum_j w_j f_j / s_j$, so the effective weight of an
objective is $w_j / s_j$, and changing one scale changes the trade-off between
the objectives and moves the optimum with it. Scales and weights multiply, so
setting both means deciding what their product should be.

```python
"objectives": {"weights": [0.6, 0.4], "scales": [1e6, 1.0]}
```

Scales must be positive: a scale is a change of units, and nothing else. Which
direction an objective is optimized in is a separate setting, described
[below](#objective-direction).

Set `auto_scale` to estimate a scale from the first batch of evaluations
instead of stating one:

```python
"objectives": {"auto_scale": True}
```

The estimate is the weighted average of the objectives over the realizations,
combined into a single factor using the objective weights, so that the weighted
sum of the objectives starts out at a magnitude of one. It is a *single* factor
for all objectives, which preserves their relative magnitudes and therefore both
the meaning of `weights` and, unlike a per-objective scale, the location of the
optimum. Realizations that fail do not contribute.

The estimate is computed once, from the first batch, and then fixed for the rest
of the run. It *multiplies* `scales` rather than replacing it, so a configured
scale still applies on top of an estimated one.

#### Choosing the direction of an objective { #objective-direction }

`ropt` minimizes. To maximize an objective, mark it in the `maximize` field,
which is a boolean per objective and defaults to all-false:

```python
"objectives": {"weights": [0.6, 0.4], "maximize": [False, True]}
```

The sign is flipped *after* the values of the individual realizations have been
combined, and never on the values themselves. This matters when a
[function estimator](function_estimators.md) produces a spread rather than an
average: negating the inputs of a standard deviation leaves it unchanged, so
asking to maximize it would quietly have minimized it instead. Negating the
combined value is correct whatever produced it.

What you see in the results follows from that:

- Per-realization values (`evaluations.objectives`) are scaled, never flipped.
- Combined values (`functions.objectives`) and their gradients are reported with
  the flip undone, so that they agree in sign with the values they summarize.
- `target_objective` is reported as the optimizer sees it, because it mixes
  objectives of different scales and directions and there is no single factor
  to undo. It is always a value being minimized, which is what lets results be
  compared by "lowest is best".

!!! note
    Objective and realization weights must not be negative. A negative weight
    used to act as a way to maximize; mark the objective in `maximize` instead.

### `linear_constraints` — [`LinearConstraintsConfig`][ropt.config.LinearConstraintsConfig] { #linear_constraints }

Linear constraints are defined by a set of linear equations involving the
optimization variables. The `coefficients` field is a 2D array where each row
represents a constraint and each column corresponds to a variable. The number of
rows determines the number of constraints.

The `lower_bounds` and `upper_bounds` fields specify the bounds on the
right-hand side of each constraint equation. They are broadcasted to match the
number of constraints.

- Less-than inequalities: set `lower_bounds` to $-\infty$.
- Greater-than inequalities: set `upper_bounds` to $+\infty$.
- Equality constraints: set `lower_bounds` equal to `upper_bounds`.

All three fields (`coefficients`, `lower_bounds`, `upper_bounds`) are required;
there are no defaults.

Coefficients refer to all variables, including those fixed by the `mask` field
of [`variables`](#variables). Before the constraints are passed to the
optimizer, the contribution of the fixed variables, evaluated at their initial
values, is subtracted from the bounds. Constraints that only involve fixed
variables reduce to a constant and are dropped. Dropping applies only to the
optimizer: results are checked against the full set of constraints, so a
constant that violates its bounds is reported in every result through
[`ConstraintInfo`][ropt.results.ConstraintInfo].

#### Scaling the equations { #linear-constraint-scales }

Two separate things happen to the linear constraints, in this order.

First, the variable `scales` and `offsets` are substituted into the equations.
Writing $x = s\,y + o$ turns $A$ into $A\,\mathrm{diag}(s)$ and the bounds $b$
into $b - A\,o$. This is a change of variables rather than a rescaling: it
shifts every equation and every bound by the same amount, so the distance
between them, and therefore the feasible set, is unchanged. It happens whenever
the variables are scaled, and needs no configuration.

Second, each equation is divided by its entry in the `scales` field, together
with its bounds:

```python
"linear_constraints": {..., "scales": [1e3, 1.0]}
```

Dividing an equation by a positive number leaves its solutions alone; what it
changes is the size of the numbers the optimizer works with, in the same way
that scaling the variables does. Set `auto_scale` to estimate those divisors
instead of stating them:

```python
"linear_constraints": {..., "auto_scale": True}
```

The estimate for an equation is the largest absolute value among its
coefficients and its finite bounds, which brings the largest entry of each row
to a magnitude of one. Only the columns of variables that are free count towards
it, since the fixed ones are eliminated before the optimizer sees the problem.
An equation with nothing to measure is left alone rather than divided by zero.
The estimate *multiplies* `scales`, so a configured scale still applies on top
of an estimated one.

Unlike `auto_scale` for nonlinear constraints, this is a single boolean rather
than one per equation: the rows form one matrix equation and are scaled together
or not at all.

Distances to the constraint bounds are reported unscaled, with both
steps undone.

### `nonlinear_constraints` — [`NonlinearConstraintsConfig`][ropt.config.NonlinearConstraintsConfig] { #nonlinear_constraints }

Nonlinear constraints are defined by comparing a constraint function to
right-hand-side bounds. The `lower_bounds` and `upper_bounds` fields specify
these bounds, and their length determines the number of constraint functions.
Both fields are required; there are no defaults.

The same bound conventions apply as for linear constraints: use $-\infty$ or
$+\infty$ for one-sided inequalities, and equal bounds for equality
constraints.

The constraint function values are returned by the evaluator in the same array
as the objectives (appended after them).

Only some optimization methods accept non-linear constraints; for the SciPy
backend, see the table in [`SciPyBackend`][ropt.backend.scipy.SciPyBackend].

Like objectives, nonlinear constraints can optionally be processed using
[realization filters](realization_filters.md) and
[function estimators](function_estimators.md), selected by key:

- `realization_filters`: default `None` (no filter applied).
- `function_estimators`: default `"0"` (the first function estimator, which by
  default computes a weighted average of per-realization values).

#### Scaling constraints { #constraint-scales }

Constraints support `scales` and `auto_scale` just as
[objectives](#objective-scales) do, with two differences.

The bounds are scaled together with the constraint values, so the constraint
you configured is the constraint that is solved. Scales are positive, so the
bounds keep their order.

Constraints have no `maximize` field: a constraint is a bound to satisfy, not a
quantity to optimize.

Auto-scaling estimates a *separate* factor for each constraint, rather than the
single shared factor used for objectives, because constraints are independent
conditions rather than terms of one weighted sum. For the same reason
`auto_scale` is a boolean *per constraint*, so that constraints can be scaled
independently:

```python
"nonlinear_constraints": {..., "auto_scale": [True, False]}
```

### `realizations` — [`RealizationsConfig`][ropt.config.RealizationsConfig] { #realizations }

To optimize an ensemble of functions, a set of realizations is defined. When the
optimizer requests a function value or a gradient, these are calculated for each
realization and then combined into a single value. Typically, this combination is
a weighted sum, but other methods are possible (see
[function estimators](function_estimators.md)).

The `weights` field determines the weight of each realization, and its length
defines the ensemble size (default: `[1.0]`, meaning a single realization). The
weights are automatically normalized to sum to 1 (for example, `[1, 1]` becomes
`[0.5, 0.5]`). As with the objective weights, they must not be negative and must
not all be zero.

If function evaluations for some realizations fail (for example, due to a simulation
error), the total function and gradient values can still be calculated by
excluding the missing values. The `realization_min_success` field specifies the
minimum number of successful realizations required (default: equal to the number
of realizations, meaning no failures are allowed).

!!! note
    Setting `realization_min_success` to zero allows the optimization to proceed
    even if all realizations fail. While some optimizers can handle this, most
    will treat it as if the value were one, requiring at least one successful
    realization.

### `optimizer` — [`OptimizerConfig`][ropt.config.OptimizerConfig] { #optimizer }

Workflow-level settings that control how the optimization run is managed. All
fields are optional and default to `None` (no limit) or `None` (no
redirection):

- **`max_batches`**: Limits the total number of *calls* made to the evaluation
  function. An optimizer might request a batch containing multiple function
  and/or gradient evaluations within a single call. This is particularly useful
  for managing resource usage when batches are evaluated in parallel (for example, on
  an HPC cluster), as it controls the number of sequential submission steps. The
  number of batches does not necessarily correspond directly to the number of
  optimizer iterations.

- **`max_functions`**: Sets a hard limit on the total *number* of individual
  objective function evaluations performed across all batches. Since a single
  batch can involve multiple function evaluations, this gives finer control
  over total computational effort. Note that exceeding this limit might
  cause the optimization to terminate mid-batch.

- **`output_dir`** (default: `None`): An optional output directory where the
  optimizer can store files. When `None`, no output directory is used.
- **`stdout`** (default: `None`): Redirect optimizer standard output to the
  given file. When `None`, standard output is not redirected.
- **`stderr`** (default: `None`): Redirect optimizer standard error to the given
  file. When `None`, standard error is not redirected.

### `backend` — [`BackendConfig`][ropt.config.BackendConfig] { #backend }

Selects the optimizer algorithm and provides a standardized set of common
settings that are forwarded to the backend:

- **`method`** (default: `"scipy/default"`): Selects the algorithm using a
  `"plugin/method"` string. The default uses SciPy's SLSQP optimizer.
- **`max_iterations`** (default: `None`): Maximum number of iterations. The
  exact definition depends on the optimizer backend, and not all backends
  support this setting.
- **`convergence_tolerance`** (default: `None`): Convergence tolerance used as a
  stopping criterion. The exact definition depends on the optimizer, and not all
  backends support this setting.
- **`parallel`** (default: `False`): If `True`, allows the optimizer to use
  parallelized function evaluations. Typically applies to gradient-free methods;
  not all backends support this setting.
- **`options`** (default: `None`): A dictionary or list of strings for generic
  optimizer options. The format and interpretation depend on the specific
  optimization method. These are passed straight to the backend.

```python
"backend": {
    "method": "scipy/default",
    "max_iterations": 200,
    "options": {"maxiter": 200},
}
```

Which methods a backend supports, which kinds of constraint and variable each
of them accepts, and which `options` they take, is documented by the backend
itself. For the built-in SciPy backend, see
[`SciPyBackend`][ropt.backend.scipy.SciPyBackend]. A method configured with a
constraint it cannot handle is only rejected when the run starts, with
[`UnsupportedError`][ropt.exceptions.UnsupportedError], so it is worth checking
the table before writing the rest of the configuration.

#### Running the optimizer in a separate process { #external-backend }

Prefix the method with `external/` to run the optimization algorithm in a
process of its own:

```python
"backend": {"method": "external/scipy/slsqp"}
```

`ropt` spawns a child process, creates the named backend there, and lets it
drive the optimization. The function and gradient evaluations still happen in
the original process: the child sends each set of variables back, the parent
evaluates it as usual, and the values are passed to the child. An error raised
in the child is re-raised in the parent.

This is useful when a backend cannot safely share a process with the rest of
your program — for example one that crashes the interpreter, leaks memory,
keeps state between runs, or links against native libraries that clash with
your other dependencies.

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

This has nothing to do with evaluating in parallel; for that see [Running in
Parallel](../getting_started/execution.md).

### `gradient` — [`GradientConfig`][ropt.config.GradientConfig] { #gradient }

Controls how stochastic gradients are estimated (see also [Stochastic
Gradients](gradients.md) for a deeper discussion).

Gradients are estimated using function values calculated from perturbed and
unperturbed variables. The `number_of_perturbations` field determines how many
perturbed variable sets are used (default:
[`DEFAULT_NUMBER_OF_PERTURBATIONS`][ropt.config.constants.DEFAULT_NUMBER_OF_PERTURBATIONS]
= `5`, must be at least 1).

If function evaluations for some perturbations fail, the gradient can still be
estimated as long as a minimum number succeed. The `perturbation_min_success`
field specifies this minimum (default: equal to `number_of_perturbations`).

Gradients are calculated for each realization individually and then combined. If
`number_of_perturbations` is low (or just 1), individual gradient calculations
may be unreliable. Setting `merge_realizations` to `True` (default: `False`)
directs the optimizer to combine the results of all realizations directly into a
single gradient estimate.

The `evaluation_policy` option (default: `"auto"`) controls how and when
objective functions and gradients are calculated:

- **`"auto"`**: Evaluate functions and/or gradients strictly according to the
  optimizer's requests.
- **`"speculative"`**: Evaluate the gradient whenever the objective function is
  requested, even if the optimizer hasn't explicitly asked for it. This can
  improve load balancing on HPC clusters by initiating gradient work earlier.
- **`"separate"`**: Always launch function and gradient evaluations as distinct
  operations, even if the optimizer requests both simultaneously. Useful when
  using [realization filters](realization_filters.md) that might disable
  certain realizations, as it can reduce the number of gradient evaluations
  needed based on information obtained from the function evaluations.

### `function_estimators`, `realization_filters`, `samplers`

These are lists of optimizer component configurations. Each entry configures a
plugin instance via a `method` field and an optional `options` dict.

They are referenced by key from the sections that use them (see [Sharing
optimizer components by key](#sharing-optimizer-components-by-key) above).

#### Function estimators — [`FunctionEstimatorConfig`][ropt.config.FunctionEstimatorConfig] { #function-estimators }

[Function estimators](function_estimators.md) control how objective and
constraint function values (and their gradients) are combined across
realizations. By default, a weighted average over realizations is used; function
estimators allow replacing that with a different combination method (for example,
standard deviation).

Fields:

- `method` (default: `"default/default"`): Selects the estimator plugin.
- `options` (default: `{}`): Plugin-specific options.

#### Realization filters — [`RealizationFilterConfig`][ropt.config.RealizationFilterConfig] { #realization-filters }

[Realization filters](realization_filters.md) modify the weights of individual
realizations. For example, they can select a subset of realizations by setting
the weights of the others to zero — useful for constructing risk-aware
objectives.

Fields:

- `method` (required, no default): Selects the filter plugin.
- `options` (default: `{}`): Plugin-specific options.

#### Samplers — [`SamplerConfig`][ropt.config.SamplerConfig] { #samplers }

[Samplers](gradients.md) generate perturbations added to variables for gradient
calculations. These perturbations can be deterministic or stochastic.

Fields:

- `method` (default: `"scipy/default"`): Selects the sampler plugin. The default
  draws perturbations from a standard normal distribution $N(0, 1)$.
- `options` (default: `{}`): Plugin-specific options.
- `shared` (default: `False`): If `True`, the same set of perturbed values is
  used for all realizations.

### `names`

Optional mapping from [`AxisName`][ropt.enums.AxisName] strings to tuples of
labels. These labels are used to produce human-readable multi-index DataFrames
when results are exported (see [Working with Results](results.md)).

Each key is an [`AxisName`][ropt.enums.AxisName] value that identifies a
dimension of the optimization problem:

| `AxisName` value           | Labels apply to                                      |
| -------------------------- | ---------------------------------------------------- |
| `"variable"`               | The optimization variables                           |
| `"objective"`              | The objective functions                              |
| `"nonlinear_constraint"`   | The nonlinear constraint functions                   |
| `"linear_constraint"`      | The linear constraints                               |
| `"realization"`            | The realizations in the ensemble                     |
| `"perturbation"`           | The perturbations used for gradient estimation       |

The corresponding value is a tuple of strings (or integers) whose length must
match the count of that axis. For example, with 3 variables and 2 objectives:

```python
"names": {
    "variable": ("x", "y", "z"),
    "objective": ("f0", "f1"),
}
```

You only need to provide labels for axes you want named — unlabelled axes
default to integer indices. See [Working with Results](results.md) for how
these labels appear in exported DataFrames.

## A worked example

```python
CONFIG = {
    "variables": {
        "variable_count": 5,
        "lower_bounds": -5.0,
        "upper_bounds":  5.0,
        "perturbation_magnitudes": 1e-5,
    },
    "objectives": {"weights": [1.0]},
    "realizations": {"weights": [1.0] * 10},
    "gradient": {"number_of_perturbations": 5},
    "optimizer": {"max_batches": 50},
    "backend": {
        "method": "scipy/default",
        "options": {"maxiter": 200},
    },
}
```

This configures a 5-variable problem with bounded variables, an ensemble of
10 equally-weighted realizations, 5 perturbations per gradient estimate,
SciPy's default optimizer, and a 50-batch cap.

## Full configuration schema

Expand the block below to see every field and its default value.

??? example "Fully expanded configuration (all defaults shown)"

    The example below shows every top-level section of the
    [`EnOptContext`][ropt.context.EnOptContext] configuration with all fields
    set to their default values. In practice you only need to specify the
    fields you want to override — everything else is filled in automatically.

    ```python
    from ropt.enums import BoundaryType, PerturbationType, VariableType

    CONFIG = {
        "variables": {
            "variable_count": ...,                            # required, no default
            "lower_bounds": -float("inf"),                    # default: -inf
            "upper_bounds": float("inf"),                     # default: +inf
            "types": VariableType.REAL,                       # default: "real" (continuous)
            "mask": True,                                     # default: all free
            "scales": 1.0,                                    # default: no scaling
            "offsets": 0.0,                                   # default: no offset
            "perturbation_magnitudes": 0.005,
            "perturbation_types": PerturbationType.ABSOLUTE,
            "boundary_types": BoundaryType.MIRROR_BOTH,
            "samplers": 0,                                    # default: use first sampler for all
            "seed": 1,
        },
        "objectives": {
            "weights": [1.0],                                 # default: single objective, weight 1.0
            "scales": 1.0,                                    # default: no scaling
            "auto_scale": False,                              # default: do not estimate scales
            "maximize": False,                                # default: minimize
            "realization_filters": None,                       # default: no filter
            "function_estimators": 0,                         # default: use first estimator for all
        },
        "linear_constraints": None,                           # No linear constraints
        "nonlinear_constraints": None,                        # No non-linear constraints
        "realizations": {
            "weights": [1.0],                                 # default: single realization, weight 1.0
            "realization_min_success": None,                  # default: equal to number of realizations
        },
        "optimizer": {
            "max_batches": None,                              # default: no limit
            "max_functions": None,                            # default: no limit
            "output_dir": None,                               # default: no output directory
            "stdout": None,                                   # default: discard
            "stderr": None,                                   # default: discard
        },
        "backend": {
            "method": "scipy/default",                        # default: SciPy SLSQP
            "max_iterations": None,                           # default: backend-specific
            "convergence_tolerance": None,                    # default: backend-specific
            "parallel": False,                                # default: Do not evaluate in parallel
            "options": None,                                  # default: no extra options
        },
        "gradient": {
            "number_of_perturbations": 5,
            "perturbation_min_success": None,                 # default: equal to number_of_perturbations
            "merge_realizations": False,                      # default: estimate and average gradients
            "evaluation_policy": "auto",                      # default: evaluate functions and perturbations
        },                                                    #          as needed
        "samplers": [
            {
                "method": "scipy/default",                    # default: standard normal N(0,1)
                "options": {},
                "shared": False,                              # default: Each realizations has its own
                                                              #          set of perturbations
            },
        ],
        "function_estimators": [
            {
                "method": "default/default",                  # default: weighted average
                "options": {},
            },
        ],
        "realization_filters": [],                            # default: none configured
        "names": {},                                          # default: none configured
    }
    ```

    Some sections above are set to `None` or `[]` because they are optional
    and problem-specific. When configured, their internal structure is as
    follows:

    ```python
    # linear_constraints (all fields required, no defaults):
    "linear_constraints": {
        "coefficients": ...,                      # required: 2D array (constraints × variables)
        "lower_bounds": ...,                      # required: 1D array (one per constraint)
        "upper_bounds": ...,                      # required: 1D array (one per constraint)
        "scales": 1.0,                            # default: no scaling
        "auto_scale": False,                      # default: do not estimate scales
    }

    # nonlinear_constraints (bounds are required, the rest has defaults):
    "nonlinear_constraints": {
        "lower_bounds": ...,                      # required: 1D array (one per constraint)
        "upper_bounds": ...,                      # required: 1D array (one per constraint)
        "scales": 1.0,                            # default: no scaling
        "auto_scale": False,                      # default: do not estimate scales
        "realization_filters": None,               # default: no filter
        "function_estimators": 0,                 # default: use first estimator
    }

    # realization_filters entries (method is required):
    "realization_filters": [
        {
            "method": ...,                        # required: str ("plugin/method")
            "options": {},
        },
    ]

    ```

## Where to next

- [Writing Evaluation Callbacks](../workflows/evaluation_callbacks.md) — produce the values that `ropt`
  consumes.
- [Working with Results](results.md) — read the optimization output.
- [Optimization Workflows](../workflows/workflows.md) — go beyond a single optimization run.
