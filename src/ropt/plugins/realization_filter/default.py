"""This plugin contains realization filters that are installed by default."""

from ropt.config import RealizationFilterConfig
from ropt.realization_filter.default import (
    DEFAULT_REALIZATION_FILTER_METHODS,
    DefaultRealizationFilter,
)

from ._base import RealizationFilterPlugin


class DefaultRealizationFilterPlugin(RealizationFilterPlugin):
    """Default realization filter plugin class."""

    @classmethod
    def create(cls, filter_config: RealizationFilterConfig) -> DefaultRealizationFilter:
        """Create a DefaultRealizationFilter instance.

        Args:
            filter_config: The realization filter configuration.

        Returns:
            A new `DefaultRealizationFilter`.
        """
        return DefaultRealizationFilter(filter_config)

    @classmethod
    def is_supported(cls, method: str) -> bool:  # ruff: ignore[undocumented-public-method]
        return method.lower() in DEFAULT_REALIZATION_FILTER_METHODS
