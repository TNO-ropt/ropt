"""This module defines the abstract base class for samplers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ropt.plugins.base import Plugin

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator
    from numpy.typing import NDArray

    from ropt.config.enopt import EnOptConfig


class Sampler(ABC):
    """Abstract Base Class for Sampler Implementations.

    This class defines the fundamental interface for all concrete sampler
    implementations within the `ropt` framework. Sampler plugins provide
    classes derived from `Sampler` that encapsulate the logic of specific
    sampling algorithms or strategies used to generate perturbed variable vectors
    for the optimization process.

    Instances of `Sampler` subclasses are created by their corresponding
    [`SamplerPlugin`][ropt.plugins.sampler.base.SamplerPlugin] factories.
    They are initialized with an
    [`EnOptConfig`][ropt.config.enopt.EnOptConfig] object detailing the
    optimization setup, the `sampler_index` identifying the specific sampler
    configuration to use from the config, an optional variable `mask` indicating
    which variables this sampler instance handles, and a NumPy random number
    generator (`rng`) for stochastic methods.

    The core functionality, generating samples, is performed by the
    `generate_samples` method, which must be implemented by subclasses.

    Subclasses must implement:

    - `__init__`: To accept the configuration, index, mask, and RNG.
    - `generate_samples`: To contain the sample generation logic.
    """

    def __init__(  # noqa: B027
        self,
        enopt_config: EnOptConfig,
        sampler_index: int,
        mask: NDArray[np.bool_] | None,
        rng: Generator,
    ) -> None:
        """Initialize the sampler object.

        The `samplers` field in the `enopt_config` is a tuple of sampler
        configurations ([`SamplerConfig`][ropt.config.enopt.SamplerConfig]).
        The `sampler_index` identifies which configuration from this tuple
        should be used to initialize this specific sampler instance.

        If a boolean `mask` array is provided, it indicates that this sampler
        instance is responsible for generating samples only for the subset of
        variables where the mask is `True`.

        Args:
            enopt_config:  The configuration of the optimizer.
            sampler_index: The index of the sampler configuration to use.
            mask:          Optional mask indicating variables handled by this sampler.
            rng:           A random generator object for stochastic methods.
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
        [`SamplerConfig`][ropt.config.enopt.SamplerConfig], the first dimension
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
            [`GradientConfig`][ropt.config.enopt.GradientConfig].

            Therefore, it is generally recommended that sampler implementations
            produce samples with a characteristic scale of approximately one
            (e.g., drawn from a distribution with a standard deviation of 1, or
            uniformly distributed within `[-1, 1]`). This allows the
            `perturbation_magnitudes` to directly control the effective size of
            the perturbations applied to the variables.

        Returns:
            A 3D NumPy array of sampled perturbation values.
        """


class SamplerPlugin(Plugin):
    """Abstract Base Class for Sampler Plugins (Factories).

    This class defines the interface for plugins responsible for creating
    [`Sampler`][ropt.plugins.sampler.base.Sampler] instances. These plugins
    act as factories for specific sampling algorithms or strategies.

    During plan execution, the [`PluginManager`][ropt.plugins.PluginManager]
    identifies the appropriate sampler plugin based on the configuration and
    uses its `create` class method to instantiate the actual `Sampler` object
    that will generate the perturbation samples.
    """

    @classmethod
    @abstractmethod
    def create(
        cls,
        enopt_config: EnOptConfig,
        sampler_index: int,
        mask: NDArray[np.bool_] | None,
        rng: Generator,
    ) -> Sampler:
        """Factory method to create a concrete Sampler instance.

        This abstract class method serves as a factory for creating concrete
        [`Sampler`][ropt.plugins.sampler.base.Sampler] objects. Plugin
        implementations must override this method to return an instance of their
        specific `Sampler` subclass.

        The [`PluginManager`][ropt.plugins.PluginManager] calls this method when
        an optimization step requires samples generated by this plugin.

        Args:
            enopt_config:  The main EnOpt configuration object.
            sampler_index: Index into `enopt_config.samplers` for this sampler.
            mask:          Optional boolean mask for variable subset sampling.
            rng:           NumPy random number generator instance.

        Returns:
            An initialized Sampler object ready for use.
        """
