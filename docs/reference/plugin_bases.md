# Plugin Base Classes

These abstract classes define the interface that each plugin area expects.
Implementing a plugin means subclassing the relevant base class and
registering it via a Python entry point under `ropt.plugins.<area>`. See
[Writing a Plugin](../utilities/writing_plugins.md) for a walkthrough.

::: ropt.plugins.backend
    options:
        members: []
::: ropt.plugins.backend.BackendPlugin
::: ropt.plugins.function_estimator
    options:
        members: []
::: ropt.plugins.function_estimator.FunctionEstimatorPlugin
::: ropt.plugins.realization_filter
    options:
        members: []
::: ropt.plugins.realization_filter.RealizationFilterPlugin
::: ropt.plugins.sampler
    options:
        members: []
::: ropt.plugins.sampler.SamplerPlugin
::: ropt.plugins.transforms
    options:
        members: []
::: ropt.plugins.transforms.VariableTransformPlugin

