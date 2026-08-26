# Utilities

Plugin-discovery helpers (`find_backend_plugin`, `find_sampler_plugin`,
`validate_backend_options`) live in `ropt.workflow`.

::: ropt.workflow
    options:
        members:
            - find_backend_plugin
            - find_sampler_plugin
            - validate_backend_options

## Concurrency

[`run_concurrent`][ropt.components.concurrency.run_concurrent] runs blocking
calls on dedicated threads rather than on the event loop's shared thread pool.
The simple API uses it to drive many optimizations at once; a workflow that has
to run blocking coordinators of its own can use it directly.

::: ropt.components.concurrency.run_concurrent
