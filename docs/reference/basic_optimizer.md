# Basic Optimization Workflow

[`BasicOptimizer`][ropt.workflow.BasicOptimizer] is a ready-made single-run
driver for applications that embed `ropt` and already have their own
batch-oriented evaluation infrastructure — for example dispatching a whole
ensemble of runs to an external scheduler at once. It wraps an
[`OptimizationStep`][ropt.components.compute_steps.OptimizationStep] and a
[`ResultsHandler`][ropt.components.event_handlers.ResultsHandler] into a
single class that takes a batch evaluator directly. For a Python script,
prefer the [simple API](../running/running.md) instead.

::: ropt.workflow._basic_optimizer
    options:
        members: False
::: ropt.workflow.BasicOptimizer

