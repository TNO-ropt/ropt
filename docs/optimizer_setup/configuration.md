# Configuration

Every `ropt` optimization run is described by a configuration dictionary. This
page covers what that dictionary becomes, the rules that apply to all of it, and
how the pieces fit together. For the field-by-field description of each section,
see [Configuration Sections](configuration_sections.md).

## The configuration object

The dictionary is validated into an [`EnOptContext`][ropt.context.EnOptContext],
where each top-level key becomes a field holding an instance of that section's
configuration class: `variables` becomes a
[`VariablesConfig`][ropt.config.VariablesConfig], `objectives` an
[`ObjectiveFunctionsConfig`][ropt.config.ObjectiveFunctionsConfig], and so on.
Only `variables` is required; every other section has defaults.

Three keys are different in kind. `samplers`, `realization_filters` and
`function_estimators` do not hold settings but **components** — objects built by
the plugin system, which the sections that use them refer to by key.

Most of the rules below follow from this one step. Because the dictionary is
converted once, validation and coercion happen at construction rather than
during the run; because the result is a set of objects rather than the dict you
wrote, those objects are frozen; and because components are shared rather than
copied, they are addressed by key.

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
[Pydantic](https://docs.pydantic.dev/). Inputs are automatically coerced to the
expected types when possible. For example, you can pass a `list` wherever a
`tuple` is expected, or a plain `list` of numbers wherever a NumPy array is
required — Pydantic will handle the conversion during validation.

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

Two runnable scripts select a component by position this way:
[examples/simple/realization_filter.py](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/realization_filter.py)
and
[examples/simple/function_estimator.py](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/function_estimator.py),
each of which registers a custom component and points the objectives at it.

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
  the same name. The name `"default"` cannot be used this way, since more than
  one plugin may define one; write it as `"plugin/default"`.

The plugin part corresponds to the name under which the plugin is registered
(via an entry point); the method part is any name the plugin declares in its
`methods` attribute. For example, the built-in SciPy backend is named `scipy`
and supports methods like `"default"`, `"SLSQP"`, and `"L-BFGS-B"`.

Both the plugin name and the method name are case-insensitive, so
`"SciPy/SLSQP"`, `"scipy/slsqp"`, and `"SCIPY/Slsqp"` all resolve to the
same backend.

The `backend` field accepts one further form, `"external/..."`, which runs the
named backend in a separate process; see [Running the optimizer in a separate
process](../running/parallel.md#external-backend).

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

## Naming things for output

One section describes nothing about the problem itself. `names` attaches labels
to the parts of a run — the variables, the objectives, the realizations, and the
rest — so that results come back readable:

```python
"names": {
    "variable": ("x", "y", "z"),
    "objective": ("val", "cost"),
    "realization": ("r0", "r1"),
},
```

Each key selects an axis, and each value is a tuple as long as that axis, so
three variables take three labels. Axes left unnamed keep their integer indices,
which is what a reader sees in place of a label. A key may also be the
name of a metadata entry holding array values, which labels the axis that
metadata spans.

Labels are not purely decorative. Some components use them to identify the items
they are given instead of relying on position, which lets results be matched
between runs whose ordering differs. That applies to the realization names; the
others affect only how a printed table reads.

[examples/simple/export.py](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/export.py)
names variables, objectives and realizations before exporting to a frame, and
[examples/simple/metadata.py](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/metadata.py)
names a metadata axis. The accepted axis names are listed under
[`names`](configuration_sections.md#names), and
[Working with Results](../running/results.md) shows how the labels appear in an
exported table.

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
SciPy's default optimizer, and a limit of 50 batches.

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
            "samplers": "0",                                  # default: use first sampler for all
            "seed": 1,
        },
        "objectives": {
            "weights": [1.0],                                 # default: single objective, weight 1.0
            "scales": 1.0,                                    # default: no scaling
            "offsets": 0.0,                                   # default: no offset
            "auto_scale": False,                              # default: do not estimate scales
            "maximize": False,                                # default: minimize
            "realization_filters": None,                       # default: no filter
            "function_estimators": "0",                       # default: use first estimator for all
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
            "stdout": None,                                   # default: not captured
            "stderr": None,                                   # default: not captured
        },
        "backend": {
            "method": "scipy/default",                        # default: SciPy SLSQP
            "max_iterations": None,                           # default: backend-specific
            "convergence_tolerance": None,                    # default: backend-specific
            "parallel": False,                                # default: Do not evaluate in parallel
            "verbose": None,                                  # default: silent
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
        "function_estimators": "0",                # default: use first estimator
    }

    # realization_filters entries (method is required):
    "realization_filters": [
        {
            "method": ...,                        # required: str ("plugin/method")
            "options": {},
        },
    ]

    ```
