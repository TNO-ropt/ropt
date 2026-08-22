# Plugin Manager

The [`PluginManager`][ropt.plugins.manager.PluginManager] discovers plugins
registered through Python entry points and looks them up by method name.

The module-level [`get_plugin`][ropt.plugins.manager.get_plugin] and
[`get_plugin_name`][ropt.plugins.manager.get_plugin_name] functions do the same
lookups against a shared manager that is created on first use. They are what
`ropt` itself calls, and are the convenient choice unless you need a manager of
your own.

See [Writing a Plugin](../utilities/writing_plugins.md) for how to implement and
register one.

::: ropt.plugins.Plugin
::: ropt.plugins.manager.PluginManager
::: ropt.plugins.manager.PluginType
::: ropt.plugins.manager.get_plugin
::: ropt.plugins.manager.get_plugin_name

