"""Public API for realization filter implementations.

Realization filters define how much each realization contributes to the values
used by the optimizer. In ensemble-based workflows, the optimizer evaluates
objective and constraint functions for multiple realizations; filters inspect
those per-realization results and return a weight for each realization.

**Core Interface**

All filter implementations inherit from the
[`RealizationFilter`][ropt.realization_filter.RealizationFilter] base class,
which defines the filter lifecycle (`__init__`, `init`) and the required
weighting method (`get_realization_weights`).

**Integration with Optimization**

Realization filters are accessed via an
[`EnOptContext`][ropt.context.EnOptContext] object through its
`realization_filters` field, a tuple of realization filter instances. Filters
are instantiated either directly as objects or via
[`RealizationFilterConfig`][ropt.config.RealizationFilterConfig] objects, which
are used by the plugin system to create instances based on the configured
method string (e.g., `"sort"` or `"cvar"`).

**Built-in and Custom Filters**

The
[`DefaultRealizationFilter`][ropt.realization_filter.default.DefaultRealizationFilter]
class provides commonly used weighting strategies, including:

- `sort`: Assign weights based on ranking objective and constraint values.
- `cvar`: Assign weights based on a Conditional Value-at-Risk selection.

Users can implement custom filters by subclassing `RealizationFilter`. Those
subclasses can be instantiated directly and passed into an
[`EnOptContext`][ropt.context.EnOptContext] object through its
`realization_filters` field. Registering a custom filter with the plugin
system is optional and only required when the filter should be selected and
configured via `RealizationFilterConfig` objects instead of being instantiated
explicitly by the user.
"""

from ._base import RealizationFilter

__all__ = [
    "RealizationFilter",
]
