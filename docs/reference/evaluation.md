# Evaluator Classes

The `ropt.evaluation` module defines the data structures exchanged between `ropt`
and user-provided evaluation functions: an input context describing which rows
must be evaluated, an output container for the objective and constraint values,
and the protocol that user callables must follow.

For detailed usage, including examples of handling inactive rows and partial
failures, see [Writing Evaluation Callbacks](../usage/evaluation_callbacks.md).
For higher-level `Evaluator` *classes* used by the workflow framework, see
[Workflow Evaluator Classes](workflow_evaluators.md).

::: ropt.evaluation.EvaluationBatchContext
::: ropt.evaluation.EvaluationBatchResult
::: ropt.evaluation.EvaluationBatchCallback

