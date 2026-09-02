# Utilities

Helpers for scripts and applications that use `ropt` live in
[`ropt.utils`](utils.md). This page covers the lower-level component utilities.

## Concurrency

[`run_concurrent`][ropt.components.concurrency.run_concurrent] runs blocking
calls on dedicated threads rather than on the event loop's shared thread pool.
The simple API uses it to drive many optimizations at once; a workflow that has
to run blocking coordinators of its own can use it directly.

::: ropt.components.concurrency.run_concurrent
