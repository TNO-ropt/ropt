# Compute Steps

A [`ComputeStep`][ropt.components.compute_steps.ComputeStep] is an executable
unit of work among `ropt`'s workflow components. Two implementations ship with `ropt`:
[`OptimizationStep`][ropt.components.compute_steps.OptimizationStep] runs an
optimization algorithm, and
[`EvaluationStep`][ropt.components.compute_steps.EvaluationStep] runs a
single ensemble evaluation.

See [Optimization Workflows](../workflows/workflows.md) for usage.

::: ropt.components.compute_steps.ComputeStep
::: ropt.components.compute_steps.EvaluationStep
::: ropt.components.compute_steps.OptimizationStep

