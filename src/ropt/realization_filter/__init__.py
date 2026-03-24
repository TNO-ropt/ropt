"""Provides plugin functionality for adding realization filters.

Realization filters are used by the optimization process to determine how the
results from a set of realizations should be weighted when evaluating the
overall objective and constraint functions. This module allows for the extension
of `ropt` with custom realization filtering strategies.
"""

from ._base import RealizationFilter

__all__ = [
    "RealizationFilter",
]
