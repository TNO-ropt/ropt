"""This module defines the abstract base class for samplers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator
    from numpy.typing import NDArray

    from ropt.config._sampler_config import SamplerConfig
    from ropt.context import EnOptContext


class Sampler(ABC):
    """Abstract Base Class for Sampler Implementations.

    This class defines the fundamental interface for all concrete sampler
    implementations within the `ropt` framework. Samplers provide classes
    derived from `Sampler` that encapsulate the logic of specific sampling
    algorithms or strategies used to generate perturbed variable vectors for the
    optimization process.

    The core functionality, generating samples, is performed by the
    `generate_samples` method, which must be implemented by subclasses.

    Subclasses must implement a `generate_samples` that contains the sample
    generation logic.
    """

    @abstractmethod
    def __init__(self, sampler_config: SamplerConfig) -> None:
        """Initialize the sampler object.

        Args:
            sampler_config: The configuration object containing settings for this sampler.
        """

    @abstractmethod
    def init(
        self, context: EnOptContext, mask: NDArray[np.bool_] | None, rng: Generator
    ) -> None:
        """Initialize the sampler object.

        Sets the internal state of the sampler, including the variable mask and
        random number generator.

        Args:
            context: The main EnOpt context object.
            mask:    Optional boolean mask for variable subset sampling.
            rng:     NumPy random number generator instance.
        """

    @abstractmethod
    def generate_samples(self) -> NDArray[np.float64]:
        """Generate and return an array of sampled perturbation values.

        This method must return a three-dimensional NumPy array containing the
        generated perturbation samples. The shape of the array should be
        `(n_realizations, n_perturbations, n_variables)`, where:

        - `n_realizations` is the number of realizations in the ensemble.
        - `n_perturbations` is the number of perturbations requested.
        - `n_variables` is the total number of optimization variables.

        If the `shared` flag is `True` in the associated
        [`SamplerConfig`][ropt.config.SamplerConfig], the first dimension
        (realizations) should have a size of 1. The framework will broadcast
        these shared samples across all realizations.

        If a boolean `mask` was provided during initialization, this sampler
        instance is responsible only for a subset of variables (where the mask
        is `True`). The returned array must still have the full `n_variables`
        size along the last axis. However, values corresponding to variables
        *not* handled by this sampler (where the mask is `False`) must be zero.

        Note: Sample Scaling and Perturbation Magnitudes
            The generated samples represent *unscaled* perturbations. During the
            gradient estimation process, these samples will be multiplied element-wise
            by the `perturbation_magnitudes` defined in the
            [`GradientConfig`][ropt.config.GradientConfig].

            Therefore, it is generally recommended that sampler implementations
            produce samples with a characteristic scale of approximately one
            (e.g., drawn from a distribution with a standard deviation of 1, or
            uniformly distributed within `[-1, 1]`). This allows the
            `perturbation_magnitudes` to directly control the effective size of
            the perturbations applied to the variables.

        Returns:
            A 3D NumPy array of sampled perturbation values.
        """
