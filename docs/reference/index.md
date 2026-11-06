# Reference

Direct links to configuration fields, API arguments and usage details.
For runnable programs by task, see the [Examples index](../getting_started/examples.md).

## Configuration

| Look up | Reference | Usage |
| --- | --- | --- |
| Configuration fields and defaults | [Configuration Reference](../optimizer_setup/configuration_sections.md) | [Configuration rules](../optimizer_setup/configuration.md) |
| All defaults in one dictionary | [Full configuration schema](../optimizer_setup/configuration_sections.md#full-configuration-schema) | [Worked configuration example](../optimizer_setup/configuration.md#a-worked-example) |
| Built-in method strings and options | [Built-in methods](../optimizer_setup/configuration_sections.md#built-in-methods) | [Optimization Choices](../optimizer_setup/choices.md) |
| Configuration classes | [Class attributes](config.md) | [Validation and broadcasting](../optimizer_setup/configuration.md#rules-that-apply-everywhere) |

## Running

| Look up | Reference | Usage |
| --- | --- | --- |
| Starting an optimization | [`optimize`][ropt.optimize] | [Running Optimizations](../running/running.md) |
| Evaluating points without optimizing | [`evaluate`][ropt.evaluate], [`evaluate_batch`][ropt.evaluate_batch] | [Evaluating without optimizing](../running/running.md#evaluating-without-optimizing) |
| Evaluation inputs and return values | [`EvaluationFunctionContext`][ropt.EvaluationFunctionContext], [`EvaluationFunctionResult`][ropt.EvaluationFunctionResult] | [Evaluation function contract and array shapes](../running/running.md#the-evaluation-function) |
| Sessions and worker pools | [`session`][ropt.session], [`Session`][ropt.Session], [`WorkerPool`][ropt.WorkerPool] | [Sessions](../running/sessions.md), [parallel evaluation](../running/parallel.md) |
| Running other work or an external program | [`run`][ropt.run], [`run_program`][ropt.run_program] | [Running External Programs](../running/programs.md) |
| Progress callbacks and stopping early | [`ResultCallback`][ropt.ResultCallback], [`aborting`][ropt.aborting] | [Stopping from a callback](../running/running.md#stopping-early-from-the-callback), [aborting from outside](../running/multiple_runs.md#stopping-from-outside) |
| Status of individual evaluations | [`StatusCallback`][ropt.StatusCallback], [`EvaluationStatus`][ropt.enums.EvaluationStatus] | [Reporting the status of evaluations](../running/running.md#reporting-the-status-of-evaluations) |
| Reusing starting-point evaluations | [`optimize` arguments][ropt.optimize] | [Reusing Results at the Starting Point](../running/initial_values.md) |

## Results and failures

| Look up | Reference | Usage |
| --- | --- | --- |
| Returned results | [`OptimizationResult`][ropt.OptimizationResult], [`EvaluationResult`][ropt.EvaluationResult] | [What an optimization returns](../running/running.md#the-result) |
| Function and gradient result fields | [Field tables](../results/results.md#what-each-field-holds), [result classes](results.md) | [Accessing result data](../results/results.md#accessing-result-data), [scaled results](../results/results.md#scaling-of-results) |
| Collecting results with handlers | [Handler API](api.md#handlers) | [Built-in handlers](../results/handlers.md#built-in-handlers) |
| Table columns and exports | [Available table fields](../results/handlers.md#available-fields), [column dictionaries to copy](../results/handlers.md#column-dicts-to-copy) | [Exporting results](../results/results.md#exporting-results) |
| Events and event metadata | [Event API](events.md) | [Passing values from event handlers](../running/running.md#passing-values-to-evaluations-from-event-handlers) |
| Exit codes and exceptions | [Exit-code table](../results/exit_codes.md), [exception classes](exceptions.md) | [Failures and Aborts](../running/failures.md) |

## Utilities and extensions

| Look up | Reference | Usage |
| --- | --- | --- |
| Utilities and enums | [Utility functions](utils.md), [enumerations](enums.md) | [Variable scaling](../optimizer_setup/variables.md#scaling) |
| Optimizer component interfaces | [Backends](backend.md), [samplers](sampler.md), [function estimators](function_estimator.md), [realization filters](realization_filter.md) | [Writing a Plugin](../advanced/writing_plugins.md) |
| Plugin registration and discovery | [Plugin system API](plugin_manager.md) | [Plugin Discovery](../advanced/plugin_discovery.md) |
