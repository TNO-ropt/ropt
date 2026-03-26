"""This module implements the SciPy sampler plugin."""

from ropt.config import SamplerConfig
from ropt.sampler.scipy import SCIPY_SAMPLER_SUPPORTED_METHODS, SciPySampler

from ._base import SamplerPlugin


class SciPySamplerPlugin(SamplerPlugin):
    """Default sampler plugin class."""

    @classmethod
    def create(
        cls,
        sampler_config: SamplerConfig,
    ) -> SciPySampler:
        """Initialize the sampler plugin.

        See the [ropt.plugins.sampler.SamplerPlugin][] abstract base class.

        # noqa
        """  # noqa: DOC201
        return SciPySampler(sampler_config)

    @classmethod
    def is_supported(cls, method: str) -> bool:
        """Check if a method is supported.

        See the [ropt.plugins.base.Plugin][] abstract base class.

        # noqa
        """  # noqa: DOC201
        return method.lower() in (SCIPY_SAMPLER_SUPPORTED_METHODS | {"default"})
