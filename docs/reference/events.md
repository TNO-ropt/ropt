# Optimization Events

Compute steps emit [`EnOptEvent`][ropt.events.EnOptEvent] objects at lifecycle
milestones. Event handlers
([`ResultsHandler`][ropt.components.event_handlers.ResultsHandler],
[`CallbackHandler`][ropt.components.event_handlers.CallbackHandler], etc.) consume
these events to track progress, store results, or trigger user logic.

See [Working with Results](../usage/results.md) for an example of subscribing to
events, and [`EnOptEventType`][ropt.enums.EnOptEventType] for the available
event types.

::: ropt.events

