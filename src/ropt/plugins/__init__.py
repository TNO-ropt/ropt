"""The `plugins` module facilitates the integration of `ropt` plugins.

The core functionality of `ropt` can be extended through plugins, which can
either be built-in, separately installed, or dynamically added at runtime.

Various types of plugins provide specific functionalities. Currently, `ropt`
supports the following plugin types:

1. [`optimizer`][ropt.plugins.optimizer]:
    Plugins that implement specific optimization methods.
2. [`sampler`][ropt.plugins.sampler]:
    Plugins responsible for generating perturbations.
3. [`realization_filter`][ropt.plugins.realization_filter]:
    Filters used to determine subsets of realizations.
4. [`function_transform`][ropt.plugins.function_transform]:
    Code used to compute objective and constraint values from realizations.
5. [`workflow`][ropt.plugins.workflow]:
    Implements workflow context objects and steps.

Plugins are managed by the [`PluginManager`][ropt.plugins.PluginManager] class.
They can be built-in, installed separately using the standard entry points
mechanism, or added dynamically using the
[`add_plugins`][ropt.plugins.PluginManager.add_plugins] method. The interface
that plugin code must adhere to in order to implement the required functionality
are define by abstract base classes.

The plugin manager object provides the
[`get_plugin`][ropt.plugins.PluginManager.get_plugin] method, which `ropt` uses
to retrieve the necessary plugin based on its type and name. Given the plugin's
type and name, this method returns a callable (either a class or a factory
function) that `ropt` uses to instantiate the plugin when needed.
"""

from ._manager import PluginManager, PluginType

__all__ = [
    "PluginType",
    "PluginManager",
]
