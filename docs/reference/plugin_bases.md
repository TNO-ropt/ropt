# Plugin Base Classes

These abstract classes define the interface that each plugin area expects.
Implementing a plugin means subclassing the relevant base class and
registering it via a Python entry point under `ropt.plugins.<area>`.

::: ropt.plugins.backend.BackendPlugin
::: ropt.plugins.function_estimator.FunctionEstimatorPlugin
::: ropt.plugins.realization_filter.RealizationFilterPlugin
::: ropt.plugins.sampler.SamplerPlugin
::: ropt.plugins.transforms.VariableTransformPlugin
::: ropt.plugins.transforms.ObjectiveTransformPlugin
::: ropt.plugins.transforms.NonlinearConstraintTransformPlugin

