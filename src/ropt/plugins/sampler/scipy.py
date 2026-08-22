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
        """Create a SciPySampler instance.

        Args:
            sampler_config: The sampler configuration.

        Returns:
            A new `SciPySampler`.
        """
        return SciPySampler(sampler_config)

    @classmethod
    def is_supported(cls, method: str) -> bool:  # ruff: ignore[undocumented-public-method]
        return method.lower() in (SCIPY_SAMPLER_SUPPORTED_METHODS | {"default"})
