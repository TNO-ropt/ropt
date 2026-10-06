# Optimization API

::: ropt
    options:
        members: []

The remaining names imported from `ropt` are documented on their own pages:
[`ExitCode`][ropt.enums.ExitCode];
[`Results`][ropt.results.Results],
[`FunctionResults`][ropt.results.FunctionResults],
[`GradientResults`][ropt.results.GradientResults],
[`results_to_pandas`][ropt.results.results_to_pandas] and
[`results_to_polars`][ropt.results.results_to_polars];
and [`RoptError`][ropt.exceptions.RoptError] with
[`WorkflowError`][ropt.exceptions.WorkflowError],
[`ExecutionError`][ropt.exceptions.ExecutionError],
[`UnsupportedError`][ropt.exceptions.UnsupportedError],
[`AbortedError`][ropt.exceptions.AbortedError] and
[`RunsFailedError`][ropt.exceptions.RunsFailedError].

## Running optimizations

::: ropt.optimize
::: ropt.optimize_many

## Evaluating without optimizing

::: ropt.evaluate
::: ropt.evaluate_batch

## Evaluation functions

::: ropt.EvaluationFunctionContext
::: ropt.EvaluationFunctionResult

## Sessions and pools

::: ropt.session
::: ropt.Session
::: ropt.WorkerPool

## Handlers

::: ropt.EventHandler
::: ropt.ResultsHandler
::: ropt.HistoryHandler
::: ropt.DataFrameHandler

## Result objects

::: ropt.OptimizationResult
::: ropt.EvaluationResult

## Callback types

::: ropt.EvaluationFunction
::: ropt.ReportCallback
