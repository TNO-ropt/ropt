# Composing custom runs

The `ropt.simple.compose` module holds the building blocks for authoring custom
runs on the high-level session. They let a custom compute step, or a launcher
that starts several runs at once, plug into the same background session,
executor and shared handlers that [`optimize`][ropt.simple.optimize] and
[`optimize_many`][ropt.simple.optimize_many] use.

These names are deliberately kept out of the `ropt.simple` namespace: end users
do not need them, and importing from `ropt.simple.compose` marks the code as
advanced use.

## Reading the open block

::: ropt.simple.compose.current_executor
::: ropt.simple.compose.current_handlers

## Running work on the session

::: ropt.simple.compose.run_step
::: ropt.simple.compose.gather_shared

## Shared handlers

::: ropt.simple.compose.HandlerScope
