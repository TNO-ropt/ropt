"""The `plugins` module facilitates the integration of `ropt` plugins.

The core functionality of `ropt` can be extended through plugins, which can
either be builtin, separately installed, or dynamically added at runtime.
Currently, `ropt` supports the following plugin types to implement specific
types of functionality:

[`optimizer`][ropt.plugins.optimizer]:
: Plugins that implement specific optimization methods. The builtin
  [`scipy`][ropt.plugins.optimizer.scipy.SciPyOptimizer] plugin utilizes the
  [`scipy.optimize`](https://docs.scipy.org/doc/scipy/reference/optimize.html)
module to implement various optimization algorithms.

[`sampler`][ropt.plugins.sampler]:
: Plugins responsible for generating perturbations for estimating gradients.
  The builtin [`scipy`][ropt.plugins.sampler.scipy.SciPySampler] plugin is based
  on [`scipy.stats`](https://docs.scipy.org/doc/scipy/reference/stats.html),
  providing various sampling methods.

[`realization_filter`][ropt.plugins.realization_filter]:
: These plugins implement filters for selecting a sub-set of realizations used
  in calculating objective or constraint functions and their gradients. The
  included
  [`default`][ropt.plugins.realization_filter.default.DefaultRealizationFilter]
  plugin provides filters based on ranking and for CVaR optimization.

[`function_estimator`][ropt.plugins.function_estimator]:
: These plugins calculate the final objective and gradient from sets of
  objectives or constraints and their gradients for individual realizations. The
  included
  [`default`][ropt.plugins.function_estimator.default.DefaultFunctionEstimator]
  plugin supports objectives defined by the mean or standard deviation of these
  values.

[`plan`][ropt.plugins.plan]:
: Plan plugins implement the objects that execute an optimization plan.
  The built-in [`default`][ropt.plugins.plan.default.DefaultPlanHandlerPlugin]
  handler and [`default`][ropt.plugins.plan.default.DefaultPlanStepPlugin] step
  plugins offer a full set of optimization plan objects for executing complex
  optimization plans.

Plugins are managed by the [`PluginManager`][ropt.plugins.PluginManager] class.
This class is used to retrieve plugin objects that derive from an abstract
base class defining the required functionality for each plugin type:

1. [`OptimizerPlugin`][ropt.plugins.optimizer.base.OptimizerPlugin]:
    Abstract base class for optimizer plugins.
2. [`SamplerPlugin`][ropt.plugins.sampler.base.SamplerPlugin]:
    Abstract base class for sampler plugins.
3. [`RealizationFilterPlugin`][ropt.plugins.realization_filter.base.RealizationFilterPlugin]:
    Abstract base class for realization filter plugins.
4. [`FunctionEstimatorPlugin`][ropt.plugins.function_estimator.base.FunctionEstimatorPlugin]:
    Abstract base class for function estimator plugins.
5. [`PlanHandlerPlugin`][ropt.plugins.plan.base.PlanHandlerPlugin]:
    Abstract base class for optimization plan object plugins.
5. [`PlanStepPlugin`][ropt.plugins.plan.base.PlanStepPlugin]:
    Abstract base class for optimization plan object plugins.

Plugins can be built-in, installed separately using the standard entry points
mechanism, or added dynamically using the
[`add_plugin`][ropt.plugins.PluginManager.add_plugin] method.

The plugin manager object provides the
[`get_plugin`][ropt.plugins.PluginManager.get_plugin] method, which `ropt` uses
to retrieve the necessary plugin based on its type and name. Given the plugin's
type and name, this method returns a callable (either a class or a factory
function) that `ropt` uses to instantiate the plugin when needed.

Info: Plugin and method names
  Plugins are registered by name by plugin manager objects. Plugins may
  implement multiple methods, each of which should also be identified by a name.
  [`PluginManager.get_plugin`][ropt.plugins.PluginManager.get_plugin] and
  [`PluginManager.is_supported`][ropt.plugins.PluginManager.is_supported] accept
  method names and will search through the available plugins to find the correct
  plugin code. To refer to a method _method-name_ of a given plugin
  _plugin-name_, a string in the form "_plugin-name/method-name_" can be used
  instead. In this case, the plugin manager will not search through all plugins
  for the requested method but will only inquire with the plugin _plugin-name_.
  By convention, using "default" for the method name in such a string will
  select the default method of the plugin.
"""

from ._manager import PluginManager, PluginType
from .base import Plugin

__all__ = [
    "Plugin",
    "PluginManager",
    "PluginType",
]
