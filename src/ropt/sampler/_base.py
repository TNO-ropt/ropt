"""Abstract base class for sampler implementations.

Samplers generate perturbation values for optimization variables during
gradient estimation. This module defines the interface that all concrete
sampler implementations must follow.
"""

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
    """Abstract base class for sampler implementations.

    All concrete sampler implementations must inherit from this class and
    implement the required lifecycle and sample-generation methods. Samplers
    are responsible for generating perturbation values that are applied to
    optimization variables when estimating gradients.

    **Lifecycle**

    1. Instantiation via `__init__`: Called by the plugin system with a
       configuration object.
    2. Setup via `init`: Called once per optimization workflow with the
       [`EnOptContext`][ropt.context.EnOptContext], a variable mask, and a
       random number generator.
    3. Sampling via `generate_samples`: Called repeatedly during optimization
       whenever perturbed variable vectors are needed.

    Subclasses must implement:

    - `__init__`: Stores sampler configuration and performs lightweight setup.
    - `init`: Receives context-dependent inputs for workflow-specific setup.
    - `generate_samples`: Returns perturbation samples with the expected shape
      and masking semantics.
    """

    @abstractmethod
    def __init__(self, sampler_config: SamplerConfig) -> None:
        """Create a new sampler instance.

        Called during instantiation. Subclasses should store the configuration
        and perform any lightweight initialization. Validation and
        context-dependent setup should usually be deferred to `init`.

        Args:
            sampler_config: Configuration object specifying the sampler method
                and any method-specific options.
        """

    @abstractmethod
    def init(
        self, context: EnOptContext, mask: NDArray[np.bool_] | None, rng: Generator
    ) -> None:
        """Finalize initialization after the optimization context is known.

        Called once at the start of each optimization workflow, after all
        configuration is finalized. Use this method to store the active
        context, receive the variable subset handled by this sampler, and
        initialize random-state dependent internals.

        Args:
            context: The main EnOpt context object.
            mask: Optional boolean mask selecting the variables handled by this
                sampler. If `None`, the sampler is responsible for all
                variables.
            rng: NumPy random number generator instance for stochastic
                sampling methods.
        """

    @abstractmethod
    def generate_samples(self) -> NDArray[np.float64]:
        """Generate perturbation samples for optimization variables.

        Returns a three-dimensional NumPy array with shape
        `(n_realizations, n_perturbations, n_variables)`, where:

        - `n_realizations` is the number of realizations in the ensemble.
        - `n_perturbations` is the number of perturbations requested.
        - `n_variables` is the total number of optimization variables.

        If the `shared` flag is `True` in the associated
        [`SamplerConfig`][ropt.config.SamplerConfig], the first dimension
        still has size `n_realizations`. Implementations may internally
        generate a single realization of samples and broadcast that internally
        before returning.

        If a boolean `mask` was provided during initialization, this sampler
        instance is responsible only for a subset of variables (where the mask
        is `True`). The returned array must still have the full `n_variables`
        size along the last axis. However, values corresponding to variables
        *not* handled by this sampler (where the mask is `False`) must be zero.

        Note: Sample Scaling and Perturbation Magnitudes
            The generated samples represent *unscaled* perturbations. During the
            gradient estimation process, these samples are multiplied
            element-wise by the `perturbation_magnitudes` defined in the
            [`GradientConfig`][ropt.config.GradientConfig].

            Therefore, it is generally recommended that sampler implementations
            produce samples with a characteristic scale of approximately one
            (e.g., drawn from a distribution with a standard deviation of 1, or
            uniformly distributed within `[-1, 1]`). This allows the
            `perturbation_magnitudes` to directly control the effective size of
            the perturbations applied to the variables.

        Returns:
            A 3D NumPy array of perturbation values.
        """
