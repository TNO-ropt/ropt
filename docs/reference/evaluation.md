# Evaluator Classes

The `ropt.evaluation` module defines the data structures exchanged between `ropt`
and user-provided evaluation functions: an input context describing which rows
must be evaluated, an output container for the objective and constraint values,
and the protocol that user callables must follow.

For detailed usage, including examples of handling inactive rows and partial
failures, see [Writing Evaluation Callbacks](../low_level/evaluation_callbacks.md).
For higher-level `Evaluator` *classes* used by the workflow components, see
[Evaluators](components_evaluators.md).

::: ropt.evaluation.EvaluationBatchContext
::: ropt.evaluation.EvaluationBatchResult
::: ropt.evaluation.EvaluationBatchCallback

