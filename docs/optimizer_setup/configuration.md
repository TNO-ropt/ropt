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
    "variable_transforms": [...],              # optional, tuple
    "objective_transforms": [...],             # optional, tuple
    "nonlinear_constraint_transforms": [...],  # optional, tuple
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

### Index-based sharing of optimizer components

Each tuple-typed field holds the optimizer **components** of the corresponding
kind:

| Field                              | Component type                                                                  |
| ---------------------------------- | ------------------------------------------------------------------------------- |
| `realization_filters`              | [`RealizationFilter`][ropt.realization_filter.RealizationFilter]                |
| `function_estimators`              | [`FunctionEstimator`][ropt.function_estimator.FunctionEstimator]                |
| `samplers`                         | [`Sampler`][ropt.sampler.Sampler]                                               |
| `variable_transforms`              | [`VariableTransform`][ropt.transforms.VariableTransform]                        |
| `objective_transforms`             | [`ObjectiveTransform`][ropt.transforms.ObjectiveTransform]                      |
| `nonlinear_constraint_transforms`  | [`NonlinearConstraintTransform`][ropt.transforms.NonlinearConstraintTransform]  |

You usually specify each component with a small config dict; see
[Providing optimizer components](#providing-optimizer-components) below.

Other config sections refer to the entries in these tuples by integer index.
For example, [`VariablesConfig`][ropt.config.VariablesConfig] has a `samplers`
field that is an integer array indexing into `EnOptContext.samplers`:

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

Use all zeros to share a single component across all elements;
thanks to broadcasting, a single `0` (the default) is sufficient.

For optional fields like `realization_filters`, an index of `-1` (the default)
or any other out-of-range value leaves the corresponding element unfiltered.

The three transform tuples are the exception: nothing indexes into them. Each
is an ordered [chain](transforms.md) applied to every variable, objective, or
constraint, in order towards the optimizer domain and in reverse coming back.

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

The same pattern applies to `backend`, `function_estimators`,
`realization_filters`, and the three transform fields. You can mix these forms
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

Once built, an [`EnOptContext`][ropt.context.EnOptContext] is frozen. To
change settings, build a new context from a modified dict.

!!! warning

    `EnOptContext` objects are immutable after construction. Do not try to
    serialize and round-trip them (for example, to/from JSON). Some parameters
    are transformed during construction in a way that cannot be undone, so
    building an `EnOptContext` from those serialized values would transform
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
[`samplers`](#function_estimators-realization_filters-samplers-transforms)
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

### `objectives` — [`ObjectiveFunctionsConfig`][ropt.config.ObjectiveFunctionsConfig] { #objectives }

`ropt` supports multi-objective optimization. Multiple objectives are combined
into a single value by summing them after weighting. The `weights` field
determines the weight of each objective function, and its length defines the
number of objectives (default: `[1.0]`, meaning a single objective). The weights
are automatically normalized to sum to 1 (for example, `[1, 1]` becomes `[0.5, 0.5]`).

```python
"objectives": {"weights": [0.6, 0.4]}
```

Objective functions can optionally be processed using
[realization filters](realization_filters.md) and
[function estimators](function_estimators.md). Both fields are integer index
arrays: each entry selects an object by its position in the corresponding tuple
defined in [`EnOptContext`][ropt.context.EnOptContext].

- `realization_filters`: default `-1` (no filter applied).
- `function_estimators`: default `0` (the first function estimator). Unless
  explicitly configured otherwise, the default function estimator method is
  `"default/default"`, which computes a weighted average of the per-realization
  values.

An out-of-range `realization_filters` index means no filter is applied to that
objective. An out-of-range `function_estimators` index leaves that objective
unestimated, and its value is reported as `NaN`.

[Objective transforms](transforms.md) are not selected per objective: the
`objective_transforms` tuple is a chain applied to every objective.

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
[function estimators](function_estimators.md) via index arrays:

- `realization_filters`: default `-1` (no filter applied).
- `function_estimators`: default `0` (the first function estimator, which by
  default computes a weighted average of per-realization values).

The `nonlinear_constraint_transforms` tuple is a
[chain](transforms.md) applied to every constraint, not selected per
constraint.

### `realizations` — [`RealizationsConfig`][ropt.config.RealizationsConfig] { #realizations }

To optimize an ensemble of functions, a set of realizations is defined. When the
optimizer requests a function value or a gradient, these are calculated for each
realization and then combined into a single value. Typically, this combination is
a weighted sum, but other methods are possible (see
[function estimators](function_estimators.md)).

The `weights` field determines the weight of each realization, and its length
defines the ensemble size (default: `[1.0]`, meaning a single realization). The
weights are automatically normalized to sum to 1 (for example, `[1, 1]` becomes
`[0.5, 0.5]`).

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
- It needs the `cloudpickle` extra (see
  [Installation](../getting_started/installation.md#optional-extras)). Without
  it, building the context raises
  [`UnsupportedError`][ropt.exceptions.UnsupportedError].

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

### `function_estimators`, `realization_filters`, `samplers`, `transforms`

These are lists of optimizer component configurations. Each entry configures a
plugin instance via a `method` field and an optional `options` dict.

`function_estimators`, `realization_filters`, and `samplers` are referenced by
index from the sections that use them (see [Index-based sharing of optimizer
components](#index-based-sharing-of-optimizer-components) above). The three
transform tuples are not: each is an ordered [chain](transforms.md) applied to
every variable, objective, or constraint.

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

#### Transforms { #transforms }

[Transforms](transforms.md) modify values as they pass between the user's
domain and the optimizer's domain. Three types exist:

- [`VariableTransformConfig`][ropt.config.VariableTransformConfig]: transforms
  variables to the optimizer's domain (default method: `"default/default"`).
- [`ObjectiveTransformConfig`][ropt.config.ObjectiveTransformConfig]: transforms
  objective values (default method: `"default/default"`).
- [`NonlinearConstraintTransformConfig`][ropt.config.NonlinearConstraintTransformConfig]:
  transforms constraint values (default method: `"default/default"`).

Each has a `method` field, an `options` dict (default: `{}`), and an optional
boolean `mask` (default: `None`) selecting the variables, objectives or
constraints the transform applies to. `None` means all of them. The transform
decides how to honour its mask; see [Transforms](transforms.md#restricting-a-transform-to-a-subset).

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
            "perturbation_magnitudes": 0.005,
            "perturbation_types": PerturbationType.ABSOLUTE,
            "boundary_types": BoundaryType.MIRROR_BOTH,
            "samplers": 0,                                    # default: use first sampler for all
            "seed": 1,
        },
        "objectives": {
            "weights": [1.0],                                 # default: single objective, weight 1.0
            "realization_filters": -1,                        # default: no filter
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
        "variable_transforms": [],                            # default: none configured
        "objective_transforms": [],                           # default: none configured
        "nonlinear_constraint_transforms": [],                # default: none configured
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
    }

    # nonlinear_constraints (bounds are required, the rest has defaults):
    "nonlinear_constraints": {
        "lower_bounds": ...,                      # required: 1D array (one per constraint)
        "upper_bounds": ...,                      # required: 1D array (one per constraint)
        "realization_filters": -1,                # default: no filter
        "function_estimators": 0,                 # default: use first estimator
    }

    # realization_filters entries (method is required):
    "realization_filters": [
        {
            "method": ...,                        # required: str ("plugin/method")
            "options": {},
        },
    ]

    # variable_transforms entries:
    "variable_transforms": [
        {
            "method": "default/default",
            "options": {},
            "mask": None,                     # default: applies to all variables
        },
    ]

    # objective_transforms entries:
    "objective_transforms": [
        {
            "method": "default/default",
            "options": {},
            "mask": None,                     # default: applies to all objectives
        },
    ]

    # nonlinear_constraint_transforms entries:
    "nonlinear_constraint_transforms": [
        {
            "method": "default/default",
            "options": {},
            "mask": None,                     # default: applies to all constraints
        },
    ]
    ```

## Where to next

- [Writing Evaluation Callbacks](../workflows/evaluation_callbacks.md) — produce the values that `ropt`
  consumes.
- [Working with Results](results.md) — read the optimization output.
- [Optimization Workflows](../workflows/workflows.md) — go beyond a single optimization run.
