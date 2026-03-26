"""Extending `ropt` with plugins.

The `ropt.plugins` module provides the framework for extending `ropt`'s
capabilities through a plugin system. Plugins allow for the integration of
custom or third-party components, installed as separate packages.

`ropt` supports several types of plugins, each addressing a specific aspect of
the optimization workflow:

***Backend related plugins:**

These plugins are used to implement specific features of the ensemble optimizer:

- [`optimizer`][ropt.backend]: Implements optimization algorithms.
- [`sampler`][ropt.sampler]: Generates parameter perturbations, which
  are used for gradient estimation.
- [`realization_filter`][ropt.plugins.realization_filter]: Selects subsets of
  ensemble realizations for calculating objectives or constraints.
- [`function_estimator`][ropt.plugins.function_estimator]: Computes final
  objective function values and gradients from individual realization results.

These plugins are generally not directly created by the user. Instead, the
optimization algorithm will find and create the required plugins based on
information passed via the optimization configuration.

**Plugin Management and Discovery**

The [`PluginManager`][ropt.plugins.manager.PluginManager] class is central to
the plugin system. It discovers and manages available plugins. Plugins are
typically discovered automatically using Python's standard entry points
mechanism.

Each plugin type has a corresponding abstract base class that custom plugins
must inherit from:

- **Backend:**
  [`BackendPlugin`][ropt.plugins.backend.BackendPlugin]
- **Sampler:** [`SamplerPlugin`][ropt.plugins.sampler.SamplerPlugin]
- **Realization Filter:**
  [`RealizationFilterPlugin`][ropt.plugins.realization_filter.RealizationFilterPlugin]
- **Function Estimator:**
  [`FunctionEstimatorPlugin`][ropt.plugins.function_estimator.FunctionEstimatorPlugin]

**Using Plugins**

The [`PluginManager.get_plugin`][ropt.plugins.manager.PluginManager.get_plugin]
method is used internally by `ropt` to retrieve the appropriate plugin
implementation based on a specified type and method name. The
[`PluginManager.get_plugin_name`][ropt.plugins.manager.PluginManager.get_plugin_name]
method can be used to find the name of a plugin that supports a given method.

Plugins can implement multiple named methods. To request a specific method
(`method-name`) from a particular plugin (`plugin-name`), use the format
`"plugin-name/method-name"`. If only a method name is provided, the plugin
manager searches through all registered plugins (that allow discovery) for one
that supports the method. Using `"plugin-name/default"` typically selects the
primary or default method offered by that plugin, although specifying
`"default"` without a plugin name is not permitted.

Plugins retrieved by the
[`PluginManager.get_plugin`][ropt.plugins.manager.PluginManager.get_plugin]
method generally implement a `create` factory method that will be used to
instantiate the objects that implement the desired functionality. These objects
must inherit
from the base class for the corresponding plugin type:

- Workflow related plugins:
  [`ComputeStep`][ropt.workflow.compute_steps.ComputeStep],
  [`EventHandler`][ropt.workflow.event_handlers.EventHandler],
  [`Evaluator`][ropt.workflow.evaluators.Evaluator]
- Backend: [`Backend`][ropt.backend.Backend]
- Sampler: [`Sampler`][ropt.sampler.Sampler]
- Realization Filter:
  [`RealizationFilter`][ropt.realization_filter.RealizationFilter]
- Function Estimator:
  [`FunctionEstimator`][ropt.function_estimator.FunctionEstimator]

**Pre-installed Plugins Included with `ropt`**

`ropt` comes bundled with a set of pre-installed plugins:

- **Backend:** The [`scipy`][ropt.plugins.backend.scipy.SciPyBackend]
  plugin, leveraging algorithms from `scipy.optimize`, and the
  [`ExternalBackend`][ropt.plugins.backend.external.ExternalBackend] plugin,
  which is used to launch optimizers in separate processes.
- **Sampler:** The [`scipy`][ropt.plugins.sampler.scipy.SciPySampler] plugin,
  using distributions from `scipy.stats`.
- **Realization Filter:** The
  [`default`][ropt.realization_filter.default.DefaultRealizationFilter]
  plugin, offering filters based on ranking and for CVaR optimization.
- **Function Estimator:** The
  [`default`][ropt.plugins.function_estimator.default.DefaultFunctionEstimator]
  plugin, supporting objectives based on mean or standard deviation.
"""

from .base import Plugin

__all__ = [
    "Plugin",
]
