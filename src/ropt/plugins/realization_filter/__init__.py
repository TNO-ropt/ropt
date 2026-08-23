"""Plugin support for realization filters.

A realization filter sets the weight of each realization whenever objective and
constraint functions are evaluated. A
[`RealizationFilterPlugin`][ropt.plugins.realization_filter.RealizationFilterPlugin]
is a factory that creates the
[`RealizationFilter`][ropt.realization_filter.RealizationFilter] objects doing
the actual work, which the
[`PluginManager`][ropt.plugins.manager.PluginManager] discovers through the
`ropt.plugins.realization_filter` entry point group.

`ropt` ships
[`DefaultRealizationFilter`][ropt.realization_filter.default.DefaultRealizationFilter],
which provides sorting and CVaR based methods.

See [Realization Filters](../optimizer_setup/realization_filters.md) for usage,
and [Writing a Plugin](../utilities/writing_plugins.md) for a walkthrough.
"""

from ._base import RealizationFilterPlugin

__all__ = [
    "RealizationFilterPlugin",
]
