"""Abstract base class for sampler implementations.

Samplers generate perturbation values for optimization variables during
gradient estimation. This module defines the interface that all concrete
sampler implementations must follow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator
    from numpy.typing import NDArray

    from ropt.config._sampler_config import SamplerConfig
    from ropt.plugins import MethodSpec


class Sampler(ABC):
    """Abstract base class for sampler implementations.

    A sampler generates the perturbation values that are applied to the
    optimization variables when estimating gradients.

    See [Stochastic Gradients](../optimizer_setup/gradients.md).
    """

    methods: ClassVar[MethodSpec]
    """The sampling methods this class provides.

    Either a set of names, which the registry matches case-insensitively, or a
    predicate for classes that cannot enumerate them. Include `"default"` in the
    set if this class has one. See [`MethodSpec`][ropt.plugins.MethodSpec].
    """

    @abstractmethod
    def __init__(self, sampler_config: SamplerConfig) -> None:
        """Create a new sampler instance.

        Called during instantiation. Subclasses should store the configuration
        and perform any lightweight initialization. Validation and setup that
        depends on the optimization should usually be deferred to `init`.

        Args:
            sampler_config: Configuration specifying the method and its options.
        """

    @abstractmethod
    def init(
        self,
        *,
        variable_count: int,
        mask: NDArray[np.bool_] | None,
        rng: Generator,
    ) -> None:
        """Finalize initialization before the optimization starts.

        Called once at the start of each optimization workflow, after all
        configuration is finalized. A `mask` of `None` makes the sampler
        responsible for every variable.

        Args:
            variable_count: The total number of optimization variables.
            mask:           Optional mask selecting this sampler's variables.
            rng:            NumPy random number generator.
        """

    @abstractmethod
    def generate_samples(
        self, *, realization_count: int, perturbation_count: int
    ) -> NDArray[np.float64]:
        """Generate perturbation samples for optimization variables.

        Returns a three-dimensional NumPy array with shape
        `(realization_count, perturbation_count, variable_count)`, where
        `variable_count` is the number passed to `init`.

        If the `shared` flag is `True` in the associated
        [`SamplerConfig`][ropt.config.SamplerConfig], the returned samples must
        be identical across all realizations. The first dimension still has
        size `realization_count`. Implementations may generate one set of
        samples and broadcast it across realizations before returning.

        If a boolean `mask` was provided during initialization, this sampler
        instance is responsible only for a subset of variables (where the mask
        is `True`). The returned array must still have the full
        `variable_count` size along the last axis. However, values
        corresponding to variables *not* handled by this sampler (where the
        mask is `False`) must be zero.

        Note: Sample Scaling and Perturbation Magnitudes
            The generated samples represent *unscaled* perturbations. During the
            gradient estimation process, these samples are multiplied
            element-wise by the `perturbation_magnitudes` defined in the
            [`VariablesConfig`][ropt.config.VariablesConfig].

            Samples with a characteristic scale of approximately one — drawn,
            for example, from a distribution with a standard deviation of 1, or
            uniformly from `[-1, 1]` — let the
            `perturbation_magnitudes` control the effective size of the
            perturbations applied to the variables directly.

        Args:
            realization_count:  The number of realizations in the ensemble.
            perturbation_count: The number of perturbations per realization.

        Returns:
            A 3D NumPy array of perturbation values.
        """
