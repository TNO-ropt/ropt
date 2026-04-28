"""Default function estimator plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ropt.function_estimator.default import (
    DEFAULT_FUNCTION_ESTIMATOR_METHODS,
    DefaultFunctionEstimator,
)

from ._base import FunctionEstimatorPlugin

if TYPE_CHECKING:
    from ropt.config import FunctionEstimatorConfig


class DefaultFunctionEstimatorPlugin(FunctionEstimatorPlugin):
    """Default filter estimator plugin class."""

    @classmethod
    def create(  # noqa: D102
        cls, estimator_config: FunctionEstimatorConfig
    ) -> DefaultFunctionEstimator:
        return DefaultFunctionEstimator(estimator_config)

    @classmethod
    def is_supported(cls, method: str) -> bool:  # noqa: D102
        return method.lower() in DEFAULT_FUNCTION_ESTIMATOR_METHODS
