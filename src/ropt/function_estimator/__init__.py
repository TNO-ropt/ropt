"""Provides functionality for function estimators.

Function estimators are used by the optimization process to combine the results
(objective function values and gradients) from a set of realizations into a
single representative value.
"""

from ._base import FunctionEstimator

__all__ = [
    "FunctionEstimator",
]
