# Configuration Reference

Field-by-field descriptions of the configuration sections, including defaults
and restrictions. For explanations and examples, start with
[Key Concepts](key_concepts.md) and follow the links to the relevant topic
pages. For how the configuration dictionary is validated and used, see
[Configuration](configuration.md). The generated API reference is under
[Configuration Classes](../reference/config.md).

## Full configuration schema

The dictionary below lists the configuration defaults. Replace `...` with the
required number of variables. Optional constraint and filter sections are
shown separately.

??? example "Configuration defaults"

    ```python
    CONFIG = {
        "variables": {
            "variable_count": ...,                            # required, no default
            "lower_bounds": -float("inf"),                    # default: -inf
            "upper_bounds": float("inf"),                     # default: +inf
            "types": "real",                                  # default: continuous
            "mask": True,                                     # default: all free
            "scales": 1.0,                                    # default: no scaling
            "offsets": 0.0,                                   # default: no offset
            "perturbation_magnitudes": 0.005,
            "boundary_types": "mirror",
            "samplers": "0",                                  # default: use first sampler for all
            "seed": 1,
        },
        "objectives": {
            "weights": [1.0],                                 # default: single objective, weight 1.0
            "scales": 1.0,                                    # default: no scaling
            "offsets": 0.0,                                   # default: no offset
            "auto_scale": False,                              # default: do not estimate scales
            "maximize": False,                                # default: minimize
            "realization_filters": None,                      # default: no filter
            "function_estimators": "0",                       # default: use first estimator for all
        },
        "linear_constraints": None,                           # default: no linear constraints
        "nonlinear_constraints": None,                        # default: no nonlinear constraints
        "realizations": {
            "weights": [1.0],                                 # default: single realization, weight 1.0
            "realization_min_success": None,                  # default: equal to number of realizations
        },
        "optimizer": {
            "max_batches": None,                              # default: no limit
            "max_functions": None,                            # default: no limit
            "output_dir": None,                               # default: no output directory
            "stdout": None,                                   # default: not captured
            "stderr": None,                                   # default: not captured
        },
        "backend": {
            "method": "scipy/default",                        # default: SciPy SLSQP
            "max_iterations": None,                           # default: backend-specific
            "convergence_tolerance": None,                    # default: backend-specific
            "parallel": False,                                # default: do not evaluate in parallel
            "verbose": None,                                  # default: backend-specific
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
                "shared": False,                             # default: each realization has its own
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

    `types` and `boundary_types` are shown as strings.
    These are the values of [`VariableType`][ropt.enums.VariableType] and
    [`BoundaryType`][ropt.enums.BoundaryType] in [`ropt.enums`][ropt.enums],
    whose members may be written in their place.

??? example "Optional constraint and filter sections"

    Replace `...` with the problem-specific values. Fields with defaults may
    be omitted.

    ```python
    # linear_constraints (coefficients and bounds are required):
    "linear_constraints": {
        "coefficients": ...,                      # required: 2D array (constraints x variables)
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
        "realization_filters": None,              # default: no filter
        "function_estimators": "0",               # default: use first estimator
    }

    # realization_filters entries (method is required):
    "realization_filters": [
        {
            "method": ...,                        # required: str ("plugin/method")
            "options": {},
        },
    ]
    ```

## Built-in methods

Use these strings in the `method` field of the indicated section or component.
The [method-string rules](configuration.md#method-strings) also describe
unqualified names. This table covers the plugins included with `ropt`;
separately installed plugins have their own method lists.

| Section | `method` values | Options and capabilities | Usage |
| --- | --- | --- | --- |
| `backend` | `scipy/default`, `scipy/nelder-mead`, `scipy/powell`, `scipy/cg`, `scipy/bfgs`, `scipy/newton-cg`, `scipy/l-bfgs-b`, `scipy/tnc`, `scipy/cobyla`, `scipy/cobyqa`, `scipy/slsqp`, `scipy/trust-constr`, `scipy/differential_evolution` | [`SciPyBackend`][ropt.backend.scipy.SciPyBackend] | [Choosing an Optimizer](optimizer.md) |
| `samplers` | `scipy/default`, `scipy/norm`, `scipy/truncnorm`, `scipy/uniform`, `scipy/sobol`, `scipy/halton`, `scipy/lhs` | [`SciPySampler`][ropt.sampler.scipy.SciPySampler] | [Selecting a sampler](samplers.md#selecting-a-sampler) |
| `function_estimators` | `default/default`, `default/mean`, `default/stddev` | [`DefaultFunctionEstimator`][ropt.function_estimator.default.DefaultFunctionEstimator] | [Selecting an estimator](function_estimators.md#selecting-an-estimator) |
| `realization_filters` | `default/cvar-objective`, `default/cvar-constraint` | [`CVaRObjectiveOptions`][ropt.realization_filter.default.CVaRObjectiveOptions], [`CVaRConstraintOptions`][ropt.realization_filter.default.CVaRConstraintOptions] | [Selecting a filter](realization_filters.md#selecting-a-filter) |

For backends, `scipy/default` selects `scipy/slsqp`; for samplers, it selects
`scipy/norm`. The estimator `default/default` selects `default/mean`.
There is no default realization-filter method.
To run a backend in a separate process, see the
[`external/` prefix](optimizer.md#external-backend).

## `variables` { #variables }

Validated into [`VariablesConfig`][ropt.config.VariablesConfig].

Defines the decision variables for the optimization problem.

The `variable_count` field is required and determines the total number of
variables, including both free and fixed variables.

The `lower_bounds` and `upper_bounds` fields define the bounds for each
variable. They are broadcast to match the number of variables and default to
$-\infty$ and $+\infty$, respectively. Use an infinity to leave a variable
unbounded on that side.

The optional `types` field assigns a type to each variable, `"real"` for
continuous or `"integer"` for discrete. If not provided, all variables default
to continuous. Integer variables are used only by methods that support them;
in the SciPy backend, that is `differential_evolution` (see
[`SciPyBackend`][ropt.backend.scipy.SciPyBackend]). The two strings are the
values of [`VariableType`][ropt.enums.VariableType]; its members can be used
instead of the strings.

The optional `mask` field is a boolean array that indicates which variables are
free to change during optimization (default: all `True`, meaning all variables
are free). `True` means the variable is free; `False` means it is fixed.

### Variable perturbations { #variable-perturbations }

These fields configure the perturbations used for gradient estimation; see
[Stochastic Gradients](gradients.md) and [Samplers](samplers.md):

- `perturbation_magnitudes`: absolute step magnitudes in the configured variable
  units (default: `0.005`).
- `boundary_types`: how perturbations outside the bounds are handled (default:
  `"mirror"`; values of [`BoundaryType`][ropt.enums.BoundaryType]).
- `samplers`: sampler key per variable (default: `"0"`).
- `seed`: random seed (default: `1`).

### Scaling the variables { #variable-scales }

The optimizer uses $y = (x-o)/s$, where `scales` $s$ must be positive and
`offsets` $o$ default to zero. `scales` defaults to one. Both fields are
broadcast to `variable_count`; there is no automatic scaling for variables.
See [Scaling](variables.md#scaling) for choosing values and how they affect the
optimization and its results.

## `objectives` { #objectives }

Validated into
[`ObjectiveFunctionsConfig`][ropt.config.ObjectiveFunctionsConfig].

`weights` has one entry per objective and therefore sets the number of
objectives. It defaults to `[1.0]`; weights are normalized to sum to one, must
be non-negative, and must not all be zero. A zero weight disables an objective.
See [Objectives](objectives.md) for how weights combine.

Objective functions can optionally be processed using
[realization filters](realization_filters.md) and
[function estimators](function_estimators.md). Both fields select an object by
its key in the corresponding section of the configuration.

- `realization_filters`: default `None` (no filter applied).
- `function_estimators`: default `"0"` (the first function estimator). Unless
  explicitly configured otherwise, the default function estimator method is
  `"default/default"`, which computes a weighted average of the per-realization
  values.

Setting `function_estimators` to `None` leaves that objective unestimated, and
its value is reported as `NaN`. Any other key must exist; see [Sharing optimizer
components by
key](configuration.md#sharing-optimizer-components-by-key).

### Scaling objectives { #objective-scales }

`scales` is positive and defaults to `1.0` per objective. The optimizer uses
each objective divided by its scale; reported objective values are converted
back to the configured units. With `auto_scale` (default: `False`), `ropt`
estimates one additional factor from the first batch and multiplies it by the
configured scales. The estimate uses the weighted objective values after
subtracting any configured offsets. See
[Weights and scales](objectives.md#weights-and-scales-determine-the-trade-off-together)
for the effect on the optimization.

### Offsetting objectives { #objective-offsets }

The optimizer uses $(f_j-o_j)/s_j$, where `offsets` $o_j$ defaults to zero per
objective. The reported objective value has the offset restored. Offsets are
subtracted before the `auto_scale` estimate; see
[Offsetting objectives](objectives.md#offsetting-objectives) for their effect
on the optimization.

### Choosing the direction of an objective { #objective-direction }

`maximize` is a boolean per objective and defaults to `False`. `ropt` applies
the direction after combining per-realization values; see
[Maximizing instead of minimizing](objectives.md#maximizing-instead-of-minimizing).

## `linear_constraints` { #linear_constraints }

Validated into [`LinearConstraintsConfig`][ropt.config.LinearConstraintsConfig].

Linear constraints are defined by a set of linear equations involving the
optimization variables. The `coefficients` field is a 2D array where each row
represents a constraint and each column corresponds to a variable. The number of
rows determines the number of constraints.

The `lower_bounds` and `upper_bounds` fields specify bounds on the value of
each linear expression. They are broadcast to match the number of constraints.

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

### Scaling the equations { #linear-constraint-scales }

`scales` is a positive per-equation divisor (default: `1.0`) applied to the
equation and its bounds. `auto_scale` is one boolean for the whole section
(default: `False`); when enabled, it estimates an additional divisor per
equation from the largest absolute coefficient among its free variables after
variable scaling. The estimate multiplies any configured scale. Variable scales
and offsets are substituted into the equations automatically. See
[Scaling constraints](constraints.md#scaling-constraints) for how these
transformations work.

## `nonlinear_constraints` { #nonlinear_constraints }

Validated into
[`NonlinearConstraintsConfig`][ropt.config.NonlinearConstraintsConfig].

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

### Scaling constraints { #constraint-scales }

`scales` is a positive divisor per constraint (default: `1.0`), applied to
both its value and bounds. `auto_scale` is a boolean per constraint (default:
`False`); when enabled, it estimates an additional scale from the first batch
and multiplies the configured scale. Constraints have no `offsets` field.
See [Scaling constraints](constraints.md#scaling-constraints) for the behavior
and its effect.

## `realizations` { #realizations }

Validated into [`RealizationsConfig`][ropt.config.RealizationsConfig].

`weights` has one entry per realization and determines the ensemble size.
Weights default to `[1.0]`, are normalized to sum to one, must be non-negative,
and must not all be zero. `realization_min_success` defaults to the ensemble
size, so every realization must succeed. Setting it to zero allows all-failed
evaluations to produce `NaN` values, which are passed to the optimizer; use this
only with an optimizer that accepts `NaN`. See [Ensemble-Based
Optimization](../getting_started/ensemble.md) and [Failures and
Aborts](../running/failures.md).

## `optimizer` { #optimizer }

Validated into [`OptimizerConfig`][ropt.config.OptimizerConfig].

All fields are optional. `max_batches` and `max_functions` default to `None`
(no limit); `output_dir`, `stdout` and `stderr` default to `None` (no output
directory or stream capture).

- `max_batches`: Maximum number of calls to the evaluation function. A call may
  contain multiple function or gradient evaluations.
- `max_functions`: Maximum number of function evaluations across calls. The
  final batch may cause the completed count to exceed this limit.
- `output_dir`: Directory for optimizer output files. Relative `stdout` and
  `stderr` paths are resolved against it.
- `stdout`: File for the optimizer's standard output. If `stderr` is unset,
  standard error is captured there too.
- `stderr`: Optional file for the optimizer's standard error.

See [Choosing an Optimizer](optimizer.md#limiting-the-length-of-a-run)
for how the limits affect a run, and [Getting Output](../results/output.md) for
capturing optimizer output.

## `backend` { #backend }

Validated into [`BackendConfig`][ropt.config.BackendConfig].

Selects the optimization method and provides common settings passed to the
backend:

- **`method`** (default: `"scipy/default"`): Selects the algorithm using a
  `"plugin/method"` string; the default is SciPy's SLSQP method.
- **`max_iterations`** (default: `None`): Maximum iterations, if supported by
  the backend.
- **`convergence_tolerance`** (default: `None`): Backend-defined stopping
  tolerance, if supported. Its meaning depends on the method.
- **`parallel`** (default: `False`): Whether the backend may parallelize
  function evaluations, if supported.
- **`verbose`** (default: `None`): Backend-specific progress output level.
- **`options`** (default: `None`): Backend-specific options passed through.

See [Choosing an Optimizer](optimizer.md) for method selection, supported
settings, and the behavior of these options. The built-in SciPy methods are
listed in [`SciPyBackend`][ropt.backend.scipy.SciPyBackend].

## `gradient` { #gradient }

Validated into [`GradientConfig`][ropt.config.GradientConfig].

Controls how stochastic gradients are estimated (see also [Stochastic
Gradients](gradients.md) for a deeper discussion).

The `gradient` section has four fields:

- `number_of_perturbations`: positive number of perturbed variable sets
  (default: [`DEFAULT_NUMBER_OF_PERTURBATIONS`][ropt.config.constants.DEFAULT_NUMBER_OF_PERTURBATIONS]).
- `perturbation_min_success`: minimum successful perturbations (default:
  `number_of_perturbations`).
- `merge_realizations`: whether realizations are combined before estimating
  the gradient (default: `False`).
- `evaluation_policy`: when functions and gradients are evaluated (default:
  `"auto"`; also `"speculative"` or `"separate"`).

See [Stochastic Gradients](gradients.md) for how these settings affect the
estimates and evaluation work.

## `function_estimators` { #function-estimators }

Validated into
[`FunctionEstimatorConfig`][ropt.config.FunctionEstimatorConfig].

A list of component configurations, each selecting a plugin with a `method`
field and an optional `options` dict, referenced by key from the sections that
use it (see [Sharing optimizer components by
key](configuration.md#sharing-optimizer-components-by-key)).

Fields:

- `method` (default: `"default/default"`): Selects the estimator plugin.
- `options` (default: `{}`): Plugin-specific options.

See [Function Estimators](function_estimators.md) for the available methods
and how to select one.

## `realization_filters` { #realization-filters }

Validated into
[`RealizationFilterConfig`][ropt.config.RealizationFilterConfig].

A list of component configurations, each selecting a plugin with a `method`
field and an optional `options` dict, referenced by key from the sections that
use it (see [Sharing optimizer components by
key](configuration.md#sharing-optimizer-components-by-key)).

Fields:

- `method` (required, no default): Selects the filter plugin.
- `options` (default: `{}`): Plugin-specific options.

See [Realization Filters](realization_filters.md) for available filters and
how to select one.

## `samplers` { #samplers }

Validated into [`SamplerConfig`][ropt.config.SamplerConfig].

A list of component configurations, each selecting a plugin with a `method`
field and an optional `options` dict, referenced by key from
[`variables.samplers`](#variables) (see [Sharing optimizer components by
key](configuration.md#sharing-optimizer-components-by-key)).

Fields:

- `method` (default: `"scipy/default"`): Selects the sampler plugin. The default
  draws perturbations from a standard normal distribution $N(0, 1)$.
- `options` (default: `{}`): Plugin-specific options.
- `shared` (default: `False`): If `True`, the same set of perturbed values is
  used for all realizations.

See [Samplers](samplers.md) for available sampling methods and their use in
gradient estimation.

## `names`

Optional mapping from dimension names to tuples of labels. The API also calls
these dimension names axis names. The labels are used to produce multi-index
DataFrames when results are exported (see
[Working with Results](../results/results.md)). They do not change the
positional order of variables, objectives or constraints or how the optimizer
uses their values.

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

A key may also be the name of a metadata key that carries array values. In that
case, the labels name the elements of the additional dimension described under
[user-defined axes](../results/results.md#user-defined-axes).

The corresponding value is a tuple of strings (or integers) whose length must
match the number of entries in that dimension. For example, with 3 variables
and 2 objectives:

```python
"names": {
    "variable": ("x", "y", "z"),
    "objective": ("f0", "f1"),
}
```

You only need to provide labels for dimensions you want named — unlabelled
dimensions default to integer indices. See
[Working with Results](../results/results.md) for how these labels appear in
exported DataFrames.
