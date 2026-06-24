# Event Handlers

Event handlers attach to a
[`ComputeStep`][ropt.workflow.compute_steps.ComputeStep] and react to the events
it emits. [`ResultsHandler`][ropt.workflow.event_handlers.ResultsHandler] keeps
the best (or last) result,
[`HistoryHandler`][ropt.workflow.event_handlers.HistoryHandler] keeps
everything, [`TableHandler`][ropt.workflow.event_handlers.TableHandler] writes a
structured table, and
[`CallbackHandler`][ropt.workflow.event_handlers.CallbackHandler] forwards
selected events to a user callback.

See [Optimization Workflows](../usage/workflows.md) and
[Working with Results](../usage/results.md) for usage.

::: ropt.workflow.event_handlers.EventHandler
::: ropt.workflow.event_handlers.ResultsHandler
::: ropt.workflow.event_handlers.HistoryHandler
::: ropt.workflow.event_handlers.TableHandler
::: ropt.workflow.event_handlers.CallbackHandler

