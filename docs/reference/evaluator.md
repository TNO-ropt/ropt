# Evaluator Classes

The `ropt.evaluator` module defines the data structures exchanged between `ropt`
and user-provided evaluation functions: an input context describing which rows
must be evaluated, an output container for the objective and constraint values,
and the protocol that user callables must follow.

For detailed usage, including examples of handling inactive rows and partial
failures, see [Writing an Evaluator Callback](../usage/evaluator_callback.md).
For higher-level `Evaluator` *classes* used by the workflow framework, see
[Workflow Evaluator Classes](workflow_evaluators.md).

::: ropt.evaluator.EvaluatorContext
::: ropt.evaluator.EvaluatorResult
::: ropt.evaluator.EvaluatorCallback

