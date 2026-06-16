# Core Classes

The `ropt.core` package contains the low-level engines used by
the workflow framework: an ensemble evaluator that orchestrates per-realization
function calls, an ensemble optimizer that drives the chosen backend, and the
callback protocols connecting them. Most users will not interact with these
classes directly; they are exposed for plugin authors and advanced workflow
developers.

See [Optimization Workflows](../usage/workflows.md) for the higher-level
framework that wraps these engines.

::: ropt.core.EnsembleEvaluator
::: ropt.core.EnsembleOptimizer
::: ropt.core.SignalEvaluationCallback
::: ropt.core.OptimizerCallback
::: ropt.core.OptimizerCallbackResult

