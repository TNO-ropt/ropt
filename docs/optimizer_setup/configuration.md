# Configuration

Every `ropt` optimization is described by a configuration dictionary. This page
covers what that dictionary holds, the rules that apply to it, and how the
pieces fit together. For the field-by-field description of each section, see
[Configuration Reference](configuration_sections.md).

## The configuration object

The dictionary is validated when the optimization or evaluation starts, and each
top-level key becomes an instance of that section's configuration class:
`variables` becomes a
[`VariablesConfig`][ropt.config.VariablesConfig], `objectives` an
[`ObjectiveFunctionsConfig`][ropt.config.ObjectiveFunctionsConfig], and so on.
Only `variables` is required; every other section has defaults.

Three keys are different in kind. `samplers`, `realization_filters` and
`function_estimators` do not hold settings but **components** — objects built by
the plugin system, which the sections that use them refer to by key.

Most of the rules below follow from this one step. Because the dictionary is
converted once, validation and coercion happen before the optimization rather
than during it; because the result is a set of objects rather than the dict you wrote, those
objects are frozen; and because components are shared rather than copied, they
are addressed by key.

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
[examples/realization_filter.py](https://github.com/TNO-ropt/ropt/blob/main/examples/realization_filter.py)
and
[examples/function_estimator.py](https://github.com/TNO-ropt/ropt/blob/main/examples/function_estimator.py),
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
same backend. The [built-in method table](configuration_sections.md#built-in-methods)
lists the supported strings and links to their options.

The `backend` field accepts one further form, `"external/..."`, which runs the
named backend in a separate process; see [Running the optimizer in a separate
process](optimizer.md#external-backend).

### Immutability

The objects the dictionary is validated into are frozen, so a setting cannot be
changed once an optimization has started. Validation also does work that a later
change would not repeat: bounds, perturbation magnitudes and linear constraints
are scaled at that point. To change a setting, modify the dictionary and start a
new optimization.

!!! warning

    Do not serialize and round-trip the validated configuration objects (for
    example, to and from JSON). Some parameters are scaled during validation in
    a way that cannot be undone, so validating those serialized values would
    scale them again, incorrectly. NumPy arrays and plugin instances may also
    not come back unchanged from a round-trip. Persist the raw input dicts
    instead.

## Naming things for output

The [`names`](configuration_sections.md#names) section attaches labels to
variables, objectives, realizations and other items in
[exported results](../results/results.md). For example,
[examples/export.py](https://github.com/TNO-ropt/ropt/blob/main/examples/export.py)
uses:

```python
"names": {
    "variable": ("x", "y", "z"),
    "objective": ("val", "cost"),
    "realization": ("r0", "r1"),
},
```

Each key specifies what to label: `"variable"` labels the variables,
`"objective"` labels the objectives, and `"realization"` labels the
realizations. The tuple must contain one label per item, in the same order,
so three variables need three labels. Items without labels are identified by
their integer positions instead. When you export results to a table, the labels
appear in its row index by default; you can also move a dimension into the
column names. See [Working with Results](../results/results.md#exporting-to-pandas)
for examples.

You can also label the elements of an array returned as metadata. Use the
metadata entry's name as the key and provide one label per array element, as
shown in
[examples/metadata.py](https://github.com/TNO-ropt/ropt/blob/main/examples/metadata.py).

Labels do not change how the optimizer interprets values or matches reused
evaluations: those operations depend on position, not labels. Keep the ordering
consistent when reusing results between optimizations.

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

See the [full configuration schema](configuration_sections.md#full-configuration-schema)
in Configuration Reference for all fields and their defaults.
