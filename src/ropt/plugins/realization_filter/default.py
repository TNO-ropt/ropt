"""This plugin contains realization filters that are installed by default."""

from ropt.config import EnOptConfig
from ropt.realization_filter.default import (
    DEFAULT_REALIZATION_FILTER_METHODS,
    DefaultRealizationFilter,
)

from ._base import RealizationFilterPlugin


class DefaultRealizationFilterPlugin(RealizationFilterPlugin):
    """Default realization filter plugin class."""

    @classmethod
    def create(
        cls, enopt_config: EnOptConfig, filter_index: int
    ) -> DefaultRealizationFilter:
        """Initialize the realization filter plugin.

        See the [ropt.plugins.realization_filter.RealizationFilterPlugin][]
        abstract base class.

        # noqa
        """  # noqa: DOC201
        return DefaultRealizationFilter(enopt_config, filter_index)

    @classmethod
    def is_supported(cls, method: str) -> bool:
        """Check if a method is supported.

        See the [ropt.plugins.base.Plugin][] abstract base class.

        # noqa
        """  # noqa: DOC201
        return method.lower() in DEFAULT_REALIZATION_FILTER_METHODS
