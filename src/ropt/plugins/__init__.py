"""Extending `ropt` with Plugins.

The `ropt.plugins` module provides the framework for extending `ropt`'s
capabilities through a plugin system. Plugins allow for the integration of
custom or third-party components, installed as separate packages.

`ropt` supports several types of plugins, each addressing a specific aspect of
the optimization workflow:

* [`plan`][ropt.plugins.plan]: Defines components for constructing and executing
  optimization plans
  ([`PlanHandlerPlugin`][ropt.plugins.plan.base.PlanHandlerPlugin] and
  [`PlanStepPlugin`][ropt.plugins.plan.base.PlanStepPlugin]).
* [`optimizer`][ropt.plugins.optimizer]: Implements optimization algorithms.
* [`sampler`][ropt.plugins.sampler]: Generates parameter perturbations, which
  are used for gradient estimation.
* [`realization_filter`][ropt.plugins.realization_filter]: Selects subsets of
  ensemble realizations for calculating objectives or constraints.
* [`function_estimator`][ropt.plugins.function_estimator]: Computes final
  objective function values and gradients from individual realization results.

**Plugin Management and Discovery**

The [`PluginManager`][ropt.plugins.PluginManager] class is central to the plugin
system. It discovers and manages available plugins. Plugins are typically
discovered automatically using Python's standard entry points mechanism.

Each plugin type has a corresponding abstract base class that custom plugins
must inherit from:

* **Plan:** [`PlanHandlerPlugin`][ropt.plugins.plan.base.PlanHandlerPlugin] and
  [`PlanStepPlugin`][ropt.plugins.plan.base.PlanStepPlugin]
* **Optimizer:**
  [`OptimizerPlugin`][ropt.plugins.optimizer.base.OptimizerPlugin]
* **Sampler:** [`SamplerPlugin`][ropt.plugins.sampler.base.SamplerPlugin]
* **Realization Filter:**
  [`RealizationFilterPlugin`][ropt.plugins.realization_filter.base.RealizationFilterPlugin]
* **Function Estimator:**
  [`FunctionEstimatorPlugin`][ropt.plugins.function_estimator.base.FunctionEstimatorPlugin]

**Using Plugins**

The [`PluginManager.get_plugin`][ropt.plugins.PluginManager.get_plugin] method
is used internally by `ropt` to retrieve the appropriate plugin implementation
based on a specified type and method name. The
[`PluginManager.is_supported`][ropt.plugins.PluginManager.is_supported] method
can check if a specific method is available.

Plugins can implement multiple named methods. To request a specific method
(`method-name`) from a particular plugin (`plugin-name`), use the format
`"plugin-name/method-name"`. If only a method name is provided, the plugin
manager searches through all registered plugins (that allow discovery) for one
that supports the method. Using `"plugin-name/default"` typically selects the
primary or default method offered by that plugin, although specifying "default"
without a plugin name is not permitted.

Plugins retrieved by the [`PluginManager.get_plugin`][ropt.plugins.PluginManager.get_plugin]
method generally implement a `create` factory method that will be used to instantiate the objects
that implement the desired functionality. These objects must inherit from the base class for the
corresponding plugin type:

* **Plan:** [`PlanHandler`][ropt.plugins.plan.base.PlanHandler] and
  [`PlanStep`][ropt.plugins.plan.base.PlanStep]
* **Optimizer:**
  [`Optimizer`][ropt.plugins.optimizer.base.Optimizer]
* **Sampler:** [`Sampler`][ropt.plugins.sampler.base.Sampler]
* **Realization Filter:**
  [`RealizationFilter`][ropt.plugins.realization_filter.base.RealizationFilter]
* **Function Estimator:**
  [`FunctionEstimator`][ropt.plugins.function_estimator.base.FunctionEstimator]

**Pre-installed Plugins Included with `ropt`**

`ropt` comes bundled with a set of pre-installed plugins:

* **Plan:** The built-in
  [`default`][ropt.plugins.plan.default.DefaultPlanHandlerPlugin] handler and
  [`default`][ropt.plugins.plan.default.DefaultPlanStepPlugin] step plugins,
  providing components for executing complex optimization plans.
* **Optimizer:** The [`scipy`][ropt.plugins.optimizer.scipy.SciPyOptimizer]
  plugin, leveraging algorithms from `scipy.optimize`, and the
  [`ExternalOptimizer`][ropt.plugins.optimizer.external.ExternalOptimizer],
  which is used to launch optimizers in separate processes.
* **Sampler:** The [`scipy`][ropt.plugins.sampler.scipy.SciPySampler] plugin,
  using distributions from `scipy.stats`.
* **Realization Filter:** The
  [`default`][ropt.plugins.realization_filter.default.DefaultRealizationFilter]
  plugin, offering filters based on ranking and for CVaR optimization.
* **Function Estimator:** The
  [`default`][ropt.plugins.function_estimator.default.DefaultFunctionEstimator]
  plugin, supporting objectives based on mean or standard deviation.
"""

from ._manager import PluginManager, PluginType
from .base import Plugin

__all__ = [
    "Plugin",
    "PluginManager",
    "PluginType",
]
