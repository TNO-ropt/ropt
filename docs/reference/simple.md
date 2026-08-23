# Simple API

::: ropt.simple
    options:
        members: []

## Running optimizations

::: ropt.simple.optimize
::: ropt.simple.optimize_many

## Evaluating without optimizing

::: ropt.simple.evaluate
::: ropt.simple.evaluate_many

## Sessions and pools

::: ropt.simple.session
::: ropt.simple.Session
::: ropt.simple.WorkerPool
::: ropt.simple.serial_pool

## Offloading work to a pool

::: ropt.simple.offload

## Aggregating results across runs

[`Session.shared_handlers`][ropt.simple.Session.shared_handlers] builds the
group; the group itself is a `SharedHandlers` object.

::: ropt.simple.SharedHandlers

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
