# Context Class

[`EnOptContext`][ropt.context.EnOptContext] is the validated, frozen container
that holds every setting needed to execute a single optimization run. It is
typically built from a plain dict (`EnOptContext.model_validate(CONFIG)`).

For a narrative overview of all fields — including broadcasting rules,
index-based sharing of plugin instances, defaults, and worked examples — see
the [Configuration](../usage/configuration.md) user-manual page.

::: ropt.context
    options:
        members:
            - EnOptContext
