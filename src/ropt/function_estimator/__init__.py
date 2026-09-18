"""Function estimators: aggregate per-realization values into one total.

An estimator reduces the function and gradient values of the ensemble to the
single numbers the optimizer consumes.

See [Function Estimators](../optimizer_setup/function_estimators.md) for usage and
algorithm descriptions.
"""

from ._base import FunctionEstimator

__all__ = [
    "FunctionEstimator",
]
