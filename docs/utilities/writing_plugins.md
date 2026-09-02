# Writing a Plugin

A plugin makes your own optimizer, sampler, filter, estimator, or transform
selectable from a configuration by name, in the same way the built-in ones are:

```python
"backend": {"method": "my_package/my_method"}
```

A plugin is a small factory class. It answers which method names it provides,
and builds the object that does the actual work. `ropt` finds it through a
Python entry point, so nothing has to be imported or registered by hand.

!!! note

    You do not need a plugin to use your own component. Every component field
    also accepts an already-constructed object, so a `Sampler` subclass can be
    passed straight into the configuration (see [Providing optimizer
    components](../optimizer_setup/configuration.md#providing-optimizer-components)).
    Write a plugin when the component should be selectable by name — typically
    when you ship it in a package for others to configure.

## The plugin areas

There is one plugin area per component type. Each has its own entry-point group,
its own factory base class, and the base class of the object it creates:

| Entry-point group | Factory base class | Creates |
| ----------------- | ------------------ | ------- |
| `ropt.plugins.backend` | [`BackendPlugin`][ropt.plugins.backend.BackendPlugin] | [`Backend`][ropt.backend.Backend] |
| `ropt.plugins.sampler` | [`SamplerPlugin`][ropt.plugins.sampler.SamplerPlugin] | [`Sampler`][ropt.sampler.Sampler] |
| `ropt.plugins.realization_filter` | [`RealizationFilterPlugin`][ropt.plugins.realization_filter.RealizationFilterPlugin] | [`RealizationFilter`][ropt.realization_filter.RealizationFilter] |
| `ropt.plugins.function_estimator` | [`FunctionEstimatorPlugin`][ropt.plugins.function_estimator.FunctionEstimatorPlugin] | [`FunctionEstimator`][ropt.function_estimator.FunctionEstimator] |
| `ropt.plugins.variable_transform` | [`VariableTransformPlugin`][ropt.plugins.transforms.VariableTransformPlugin] | [`VariableTransform`][ropt.transforms.VariableTransform] |
| `ropt.plugins.objective_transform` | [`ObjectiveTransformPlugin`][ropt.plugins.transforms.ObjectiveTransformPlugin] | [`ObjectiveTransform`][ropt.transforms.ObjectiveTransform] |
| `ropt.plugins.nonlinear_constraint_transform` | [`NonlinearConstraintTransformPlugin`][ropt.plugins.transforms.NonlinearConstraintTransformPlugin] | [`NonlinearConstraintTransform`][ropt.transforms.NonlinearConstraintTransform] |

## What a plugin must implement

All factory classes derive from [`Plugin`][ropt.plugins.Plugin] and share the
same two-method interface:

- **`is_supported(method)`** — return `True` for every method name the plugin
  provides. It receives the method name only, without the plugin prefix. Method
  names are matched case-insensitively, so compare in lower case.
- **`create(config)`** — build and return the object. The argument is the
  validated configuration object for the area, for example a
  [`SamplerConfig`][ropt.config.SamplerConfig] for a sampler, carrying the
  `method` string and the `options` given in the configuration.

One method is optional:

- **`allows_discovery()`** — return `False` to keep the plugin from being
  matched when a configuration gives a bare method name without a plugin
  prefix. The built-in
  [`external`][ropt.backend.external.ExternalBackend] backend does this,
  because its method names belong to the backend it delegates to. The default
  is `True`.

By convention every plugin also supports the method name `"default"`, so that
`"my_package/default"` selects whatever the plugin considers its standard
choice.

## An example

A sampler that draws uniform perturbations. First the sampler itself, a
[`Sampler`][ropt.sampler.Sampler] subclass:

```python
import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from ropt.config import SamplerConfig
from ropt.context import EnOptContext
from ropt.sampler import Sampler


class UniformSampler(Sampler):
    def __init__(self, sampler_config: SamplerConfig) -> None:
        self._config = sampler_config

    def init(
        self, context: EnOptContext, mask: NDArray[np.bool_] | None, rng: Generator
    ) -> None:
        self._rng = rng

    def generate_samples(self) -> NDArray[np.float64]:
        ...
```

Then the factory that makes it selectable:

```python
from ropt.plugins.sampler import SamplerPlugin


class UniformSamplerPlugin(SamplerPlugin):
    @classmethod
    def is_supported(cls, method: str) -> bool:
        return method.lower() in {"default", "uniform"}

    @classmethod
    def create(cls, sampler_config: SamplerConfig) -> Sampler:
        return UniformSampler(sampler_config)
```

## Registering it

Declare the factory class under the entry-point group of its area, in the
`pyproject.toml` of the package that contains it:

```toml
[project.entry-points."ropt.plugins.sampler"]
my_package = "my_package.sampler:UniformSamplerPlugin"
```

The entry-point name is the plugin name used in method strings. After
installing the package, the sampler is available as `"my_package/uniform"`, or
as `"uniform"` if no other installed plugin claims that name:

```python
"samplers": [{"method": "my_package/uniform"}],
```

Check that the plugin is found with
[`get_plugin_name`][ropt.plugins.manager.get_plugin_name]:

```python
from ropt.utils import get_plugin_name

get_plugin_name("sampler", "my_package/uniform")   # "my_package"
```

Plugins are discovered once: the shared plugin manager scans the entry points
the first time a method is looked up. A package installed while the program is
running is not picked up.

## Validating options

Backends receive their method-specific settings through the `options` field of
[`BackendConfig`][ropt.config.BackendConfig], which `ropt` passes through
unchecked. A backend reports what it accepts by implementing
`validate_options`, which is what
[`validate_backend_options`][ropt.utils.validate_backend_options] calls so
that users can catch mistakes before starting a long run.

[`OptionsSchemaModel`][ropt.config.options.OptionsSchemaModel] describes the
options of each method in one place, and both validates them and generates the
documentation table for them. The built-in SciPy backend uses it; see
`SCIPY_OPTIONS_SCHEMA` in `ropt.backend.scipy` for a complete example.

## Where to next

- The interfaces in full: [Plugin Base Classes](../reference/plugin_bases.md)
  and [Plugin Manager](../reference/plugin_manager.md).
- What the built-in plugins register:
  [Default Plugins](../reference/default_plugins.md).
- Looking up and validating installed plugins:
  [Plugin Discovery](plugin_discovery.md).
- How method strings are resolved:
  [Method strings](../optimizer_setup/configuration.md#method-strings).
