"""Provides plugin functionality for adding realization filters.

Realization filters are used by the optimization process to determine how the
results from a set of realizations should be weighted when evaluating the
overall objective and constraint functions. This module allows for the extension
of `ropt` with custom realization filtering strategies.

**Core Concepts:**

* **Plugin Interface:** Realization filter plugins must inherit from the
  [`RealizationFilterPlugin`][ropt.plugins.realization_filter.base.RealizationFilterPlugin]
  base class. This class acts as a factory, defining a `create` method to
  instantiate filter objects.
* **Filter Implementation:** The actual filtering logic resides in classes that
  inherit from the
  [`RealizationFilter`][ropt.plugins.realization_filter.base.RealizationFilter]
  abstract base class. These classes are initialized with the optimization
  configuration ([`EnOptConfig`][ropt.config.enopt.EnOptConfig]) and the index
  of the specific filter configuration to use (`filter_index`). The core
  functionality is provided by the `get_realization_weights` method, which
  calculates and returns weights for each realization based on their objective
  and constraint values.
* **Discovery:** The [`PluginManager`][ropt.plugins.PluginManager] discovers
  available `RealizationFilterPlugin` implementations (typically via entry
  points) and uses them to create `RealizationFilter` instances as needed during
  plan execution.

**Built-in Realization Filter Plugins:**

The default
[`DefaultRealizationFilter`][ropt.plugins.realization_filter.default.DefaultRealizationFilter]
class provides several filtering methods, including sorting by
objective/constraint values and Conditional Value-at-Risk (CVaR) based
weighting.
"""
