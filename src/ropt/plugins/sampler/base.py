"""This module defines the abstract base class for samplers.

Samplers can be added via the plugin mechanism to implement additional ways to
generate perturbed variables. Any object that follows the
[`Sampler`][ropt.plugins.sampler.base.Sampler] abstract base class may be
installed as a plugin.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from ropt.plugins.base import Plugin

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator
    from numpy.typing import NDArray

    from ropt.config.enopt import EnOptConfig


class Sampler(ABC):
    """Abstract base class for sampler classes.

    `ropt` employs plugins to implement samplers that are called during an
    optimization workflow to generate perturbed variable vectors. Samplers
    should derive from the `Sampler` base class, which specifies the
    requirements for the class constructor (`__init__`) and also includes a
    `generate_samples` method used to generate samples used to create perturbed
    values.
    """

    @abstractmethod
    def __init__(
        self,
        enopt_config: EnOptConfig,
        sampler_index: int,
        variable_indices: Optional[NDArray[np.intc]],
        rng: Generator,
    ) -> None:
        """Initialize the sampler object.

        The `samplers` field in the
        [`enopt_config`][ropt.config.enopt.EnOptConfig] configuration used by
        the optimization is a tuple of sampler configurations (see
        [`SamplerConfig`][ropt.config.enopt.SamplerConfig]). The `sampler_index`
        field is used to identify the configuration to use to initialize this
        sampler.

        The sampler may be used for a subset of the variables. The `variable_indices`
        array lists the indices of the variables that are handled by this sampler.

        Arguments:
            enopt_config:     The configuration of the optimizer.
            sampler_index:    The index of the sampler to use.
            variable_indices: The indices of the variables to sample.
            rng:              A random generator object for use by stochastic samplers.
        """

    @abstractmethod
    def generate_samples(self) -> NDArray[np.float64]:
        """Return an array containing sampled values.

        The result should be a three-dimensional array of perturbation values.
        The variable values are stored along the last axis, for each realization
        and perturbation. The first axis indexes the realization, and the second
        axis indexes the perturbation.

        If the `shared` flag is set in the
        [`SamplerConfig`][ropt.config.enopt.SamplerConfig] configuration, the
        first dimension should have a length equal to one, since all
        realizations will use the same set of perturbations.

        The sampler may handle only a subset of the variables, as specified by
        the `variable_indices` argument of the constructor. In this case, only
        the corresponding values along the variables axis (the last axis) should
        be set, while other values should be zero.

        Note: Sample scaling
            Samples will be multiplied by the values given by the
            `perturbation_magnitudes` field in the
            [`gradients`][ropt.config.enopt.GradientConfig] section of the
            optimizer configuration. It makes therefore sense to generate
            samples that have an order of magnitude around one. For instance, by
            generating them on a `[-1, 1]` range, or with a unit standard
            deviation.

        Returns:
            The sampled values.
        """


class SamplerPlugin(Plugin):
    """Abtract base class for sampler plugins."""

    @abstractmethod
    def create(
        self,
        enopt_config: EnOptConfig,
        sampler_index: int,
        variable_indices: Optional[NDArray[np.intc]],
        rng: Generator,
    ) -> Sampler:
        """Create a sampler.

        Arguments:
            enopt_config:     The configuration of the optimizer.
            sampler_index:    The index of the sampler to use.
            variable_indices: The indices of the variables to sample.
            rng:              A random generator object for use by stochastic samplers.
        """
