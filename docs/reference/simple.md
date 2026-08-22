# Simple API

The `ropt.simple` module is the high-level, convenience interface
for running optimizations. See [Running Optimizations](../running/running.md) for a
walkthrough.

Enumerations used in the configuration and results (for example
[`ExitCode`][ropt.enums.ExitCode] and [`VariableType`][ropt.enums.VariableType])
are not part of this module; import them from [`ropt.enums`][ropt.enums].

Nothing about a run depends on where it is called from. Where its evaluations
happen is decided by the pool it is given with `pool=`, and which handlers see
its results by the `handlers=` it is given. A [`session`][ropt.simple.session]
hands out both; a run given no pool evaluates in-process. This holds wherever
the run is started from, including a thread you spawn yourself.

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
