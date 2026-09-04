# Realization Filters

A realization filter selects which realizations contribute to a function or
gradient value at each evaluation. The default provides CVaR-style tail
selection, enabling risk-aware objectives.

See [Realization Filters](../optimizer_setup/realization_filters.md) for usage.

::: ropt.realization_filter
    options:
        members: []
::: ropt.realization_filter.RealizationFilter
::: ropt.realization_filter.default.DefaultRealizationFilter
::: ropt.realization_filter.default.CVaRObjectiveOptions
::: ropt.realization_filter.default.CVaRConstraintOptions

