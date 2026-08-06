# Event Handlers

Event handlers attach to a
[`ComputeStep`][ropt.components.compute_steps.ComputeStep] and react to the events
it emits. [`ResultsHandler`][ropt.components.event_handlers.ResultsHandler] keeps
the best (or last) result,
[`HistoryHandler`][ropt.components.event_handlers.HistoryHandler] keeps
everything, [`DataFrameHandler`][ropt.components.event_handlers.DataFrameHandler]
writes a structured table,
[`CallbackHandler`][ropt.components.event_handlers.CallbackHandler] forwards
selected events to a user callback, and
[`EventForwardHandler`][ropt.components.event_handlers.EventForwardHandler]
forwards events to an
[`EventDispatcher`][ropt.components.event_handlers.EventDispatcher] for
lock-free dispatch.

The [`EventDispatcher`][ropt.components.event_handlers.EventDispatcher] fans
events out to its registered handlers from the asyncio event loop's thread, so
handlers shared across concurrent compute steps need no locking.

See [Optimization Workflows](../usage/workflows.md) and
[Working with Results](../usage/results.md) for usage.

::: ropt.components.event_handlers.EventHandler
::: ropt.components.event_handlers.ResultsHandler
::: ropt.components.event_handlers.HistoryHandler
::: ropt.components.event_handlers.DataFrameHandler
::: ropt.components.event_handlers.CallbackHandler
::: ropt.components.event_handlers.EventForwardHandler
::: ropt.components.event_handlers.EventDispatcher

