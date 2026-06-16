# Event Handlers

Event handlers attach to a [`ComputeStep`][ropt.workflow.compute_steps.ComputeStep]
and react to the events it emits.
[`Tracker`][ropt.workflow.event_handlers.Tracker] keeps the best (or last)
result, [`Store`][ropt.workflow.event_handlers.Store] keeps everything,
[`Table`][ropt.workflow.event_handlers.Table] writes a structured table, and
[`Observer`][ropt.workflow.event_handlers.Observer] forwards selected events
to a user callback.

See [Optimization Workflows](../usage/workflows.md) and
[Working with Results](../usage/results.md) for usage.

::: ropt.workflow.event_handlers.EventHandler
::: ropt.workflow.event_handlers.Tracker
::: ropt.workflow.event_handlers.Store
::: ropt.workflow.event_handlers.Table
::: ropt.workflow.event_handlers.Observer

