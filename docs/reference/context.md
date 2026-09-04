# Context Class

[`EnOptContext`][ropt.context.EnOptContext] is the validated, frozen container
that holds every setting needed to execute a single optimization run. It is
typically built from a plain dict (`EnOptContext.model_validate(CONFIG)`).

For a narrative overview of all fields — including broadcasting rules, sharing
plugin instances by key, defaults, and worked examples — see the
[Configuration](../optimizer_setup/configuration.md) user-manual page.

The scales applied to the objectives and the nonlinear constraints of a run are
read with
[`get_objective_scales`][ropt.context.EnOptContext.get_objective_scales] and
[`get_constraint_scales`][ropt.context.EnOptContext.get_constraint_scales].

::: ropt.context
    options:
        members:
            - EnOptContext
