# Plugin Manager

The [`PluginManager`][ropt.plugins.manager.PluginManager] discovers plugins
registered through Python entry points and looks them up by method name. A
lookup returns the plugin class itself, which is called with the configuration
object for its area to build the component.

The module-level [`get_plugin`][ropt.plugins.manager.get_plugin] and
[`get_plugin_name`][ropt.plugins.manager.get_plugin_name] functions do the same
lookups against one shared manager, created on first use. Every optimization
resolves against that manager and no other, so it is the one a plugin has to be
registered with to have any effect.

Plugins are normally installed, and found through their entry points. One that
cannot be, such as a class defined in a script or a notebook, is added to the
shared manager with
[`register_plugin`][ropt.plugins.manager.register_plugin].

See [Writing a Plugin](../utilities/writing_plugins.md) for how to implement and
register one.

::: ropt.plugins.MethodSpec
::: ropt.plugins.manager.PluginManager
::: ropt.plugins.manager.PluginType
::: ropt.plugins.manager.get_plugin
::: ropt.plugins.manager.get_plugin_name
::: ropt.plugins.manager.register_plugin

