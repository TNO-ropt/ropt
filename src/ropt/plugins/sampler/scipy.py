"""This module implements the SciPy sampler plugin."""

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from ropt.config import EnOptConfig
from ropt.sampler.scipy import SCIPY_SAMPLER_SUPPORTED_METHODS, SciPySampler

from ._base import SamplerPlugin


class SciPySamplerPlugin(SamplerPlugin):
    """Default sampler plugin class."""

    @classmethod
    def create(
        cls,
        enopt_config: EnOptConfig,
        sampler_index: int,
        mask: NDArray[np.bool_] | None,
        rng: Generator,
    ) -> SciPySampler:
        """Initialize the sampler plugin.

        See the [ropt.plugins.sampler.SamplerPlugin][] abstract base class.

        # noqa
        """  # noqa: DOC201
        return SciPySampler(enopt_config, sampler_index, mask, rng)

    @classmethod
    def is_supported(cls, method: str) -> bool:
        """Check if a method is supported.

        See the [ropt.plugins.base.Plugin][] abstract base class.

        # noqa
        """  # noqa: DOC201
        return method.lower() in (SCIPY_SAMPLER_SUPPORTED_METHODS | {"default"})
