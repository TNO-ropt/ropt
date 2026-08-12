# Simple API

The `ropt.simple` module is the high-level, convenience interface
for running optimizations. See [Running Optimizations](../running/running.md) for a
walkthrough.

Enumerations used in the configuration and results (for example
[`ExitCode`][ropt.enums.ExitCode] and [`VariableType`][ropt.enums.VariableType])
are not part of this module; import them from [`ropt.enums`][ropt.enums].

## Running optimizations

::: ropt.simple.optimize
::: ropt.simple.optimize_many

## Evaluating without optimizing

::: ropt.simple.evaluate
::: ropt.simple.evaluate_many

## Execution blocks

::: ropt.simple.threads
::: ropt.simple.processes
::: ropt.simple.hpc

## Offloading work to the executor

::: ropt.simple.offload
::: ropt.simple.can_offload

## Aggregating results across runs

::: ropt.simple.handlers

## Result objects

::: ropt.simple.OptimizeResult
::: ropt.simple.EvaluateResult

## Callback types

::: ropt.simple.EvaluationFunction
::: ropt.simple.ReportCallback

## Re-exported for convenience

These names are re-exported from `ropt.simple` (so simple-API code imports them
from one place), but they are the low-level classes and are documented with the
components:

- the evaluation context and result:
  [`EvaluationFunctionContext`][ropt.components.evaluators.EvaluationFunctionContext],
  [`EvaluationFunctionResult`][ropt.components.evaluators.EvaluationFunctionResult];
- the result handlers:
  [`EventHandler`][ropt.components.event_handlers.EventHandler],
  [`HistoryHandler`][ropt.components.event_handlers.HistoryHandler],
  [`ResultsHandler`][ropt.components.event_handlers.ResultsHandler],
  [`DataFrameHandler`][ropt.components.event_handlers.DataFrameHandler].
