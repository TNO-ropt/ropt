# Configuration Classes

The [`ropt.config`][ropt.config] package contains the Pydantic models that
define an optimization run. Each class corresponds to a top-level section of
the configuration dictionary consumed by
[`EnOptContext`][ropt.context.EnOptContext] and
[`BasicOptimizer`][ropt.workflow.BasicOptimizer].

For detailed descriptions of each field, including defaults, usage patterns,
and examples, see the [Configuration](../usage/configuration.md) user-manual
page.

::: ropt.config
    options:
        members:
            - VariablesConfig
            - ObjectiveFunctionsConfig
            - LinearConstraintsConfig
            - NonlinearConstraintsConfig
            - RealizationsConfig
            - OptimizerConfig
            - BackendConfig
            - GradientConfig
            - FunctionEstimatorConfig
            - RealizationFilterConfig
            - SamplerConfig
            - VariableTransformConfig
            - ObjectiveTransformConfig
            - NonlinearConstraintTransformConfig
        group_by_category: false
        show_bases: false

::: ropt.config.constants

::: ropt.config.options
