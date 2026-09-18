"""Realization filters: select which realizations contribute to a value.

A filter is applied per evaluation, to a function or a gradient. The default
provides CVaR-style tail selection, for risk-aware objectives.

See [Realization Filters](../optimizer_setup/realization_filters.md) for usage and
algorithm descriptions.
"""

from ._base import RealizationFilter

__all__ = [
    "RealizationFilter",
]
