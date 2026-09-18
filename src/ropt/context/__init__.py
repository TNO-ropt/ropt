"""The validated configuration of a single optimization run.

[`EnOptContext`][ropt.context.EnOptContext] is the frozen container holding
every setting a run needs, normally built from a plain dictionary with
`EnOptContext.model_validate(config)`. See
[Configuration](../optimizer_setup/configuration.md) for the fields,
broadcasting rules and defaults.
"""

from ._enopt_context import EnOptContext

__all__ = [
    "EnOptContext",
]
