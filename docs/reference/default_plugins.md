# Default Plugins

The classes on this page are the plugin entry points for the implementations
that ship with `ropt`. They are what gets registered with the
[`PluginManager`][ropt.plugins.manager.PluginManager] under the standard
entry-point groups.

See the [Plugin Base Classes](plugin_bases.md) reference for the interfaces.

::: ropt.backend.scipy.SciPyBackendPlugin
::: ropt.backend.external.ExternalBackendPlugin
::: ropt.function_estimator.default.DefaultFunctionEstimatorPlugin
::: ropt.realization_filter.default.DefaultRealizationFilterPlugin
::: ropt.sampler.scipy.SciPySamplerPlugin
::: ropt.transforms.default.DefaultVariableTransformPlugin
::: ropt.transforms.default.DefaultObjectiveTransformPlugin
::: ropt.transforms.default.DefaultNonlinearConstraintTransformPlugin

