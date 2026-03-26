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
    def create(
        cls, estimator_config: FunctionEstimatorConfig
    ) -> DefaultFunctionEstimator:
        """Initialize the realization filter plugin.

        See the [ropt.plugins.function_estimator.FunctionEstimatorPlugin][]
        abstract base class.

        # noqa
        """  # noqa: DOC201
        return DefaultFunctionEstimator(estimator_config)

    @classmethod
    def is_supported(cls, method: str) -> bool:
        """Check if a method is supported.

        See the [ropt.plugins.base.Plugin][] abstract base class.

        # noqa
        """  # noqa: DOC201
        return method.lower() in DEFAULT_FUNCTION_ESTIMATOR_METHODS
