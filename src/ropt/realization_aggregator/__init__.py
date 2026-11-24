"""Realization aggregators: combine per-realization values into one total.

An aggregator reduces the function and gradient values of the ensemble to the
values the optimizer consumes.

See [Realization Aggregators](../optimizer_setup/realization_aggregators.md) for usage and
algorithm descriptions.
"""

from ._base import RealizationAggregator

__all__ = [
    "RealizationAggregator",
]
