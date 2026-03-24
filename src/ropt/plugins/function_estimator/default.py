"""Default function estimator plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ropt.config import EnOptConfig, FunctionEstimatorConfig
from ropt.function_estimator.default import (
    DEFAULT_FUNCTION_ESTIMATOR_METHODS,
    DefaultFunctionEstimator,
)

from ._base import FunctionEstimatorPlugin

if TYPE_CHECKING:
    from ropt.config import EnOptConfig


class DefaultFunctionEstimatorPlugin(FunctionEstimatorPlugin):
    """Default filter estimator plugin class."""

    @classmethod
    def create(
        cls, enopt_config: EnOptConfig, estimator_index: int
    ) -> DefaultFunctionEstimator:
        """Initialize the realization filter plugin.

        See the [ropt.plugins.function_estimator.FunctionEstimatorPlugin][]
        abstract base class.

        # noqa
        """  # noqa: DOC201
        estimator_config = enopt_config.function_estimators[estimator_index]
        assert isinstance(estimator_config, FunctionEstimatorConfig)
        return DefaultFunctionEstimator(enopt_config, estimator_config)

    @classmethod
    def is_supported(cls, method: str) -> bool:
        """Check if a method is supported.

        See the [ropt.plugins.base.Plugin][] abstract base class.

        # noqa
        """  # noqa: DOC201
        return method.lower() in DEFAULT_FUNCTION_ESTIMATOR_METHODS
