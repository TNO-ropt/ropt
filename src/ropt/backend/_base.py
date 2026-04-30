"""This module defines the base class for optimization plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ropt.config import BackendConfig
    from ropt.context import EnOptContext
    from ropt.core import OptimizerCallback


class Backend(ABC):
    """Abstract Base Class for Backend Implementations.

    This class defines the fundamental interface for all concrete optimizer
    implementations within the `ropt` framework. Backend plugins provide
    classes derived from `Backend` that encapsulate the logic of specific
    optimization algorithms.

    Instances of `Backend` subclasses must be initialized with an
    [`EnOptContext`][ropt.context.EnOptContext] object detailing the optimization
    setup and an [`OptimizerCallback`][ropt.core.OptimizerCallback] function.
    The callback is crucial as it allows the optimizer to request function and
    gradient evaluations from the `ropt` core during its execution.

    The optimization process itself is initiated by calling the `start` method,
    which must be implemented by subclasses.

    Subclasses must implement:
    - `start`: To contain the main optimization loop.

    Subclasses can optionally override:
    - `allow_nan`:   To indicate if the algorithm can handle NaN function values.
    - `is_parallel`: To indicate if the algorithm may perform parallel evaluations.
    """

    @abstractmethod
    def __init__(self, backend_config: BackendConfig) -> None:
        """Initialize the SciPy optimizer backend.

        Args:
            backend_config: The configuration for the backend, containing the
                            method name and options.
        """

    @abstractmethod
    def init(
        self, context: EnOptContext, optimizer_callback: OptimizerCallback
    ) -> None:
        """Initialize the optimizer with the given context and callback.

        This abstract method must be implemented by concrete `Backend` subclasses
        to set up the optimizer with the provided optimization context and
        callback function.

        Args:
            context:            An instance of `EnOptContext` containing details
                                about the optimization problem setup.
            optimizer_callback: An instance of `OptimizerCallback` that allows
                                the optimizer to request function and gradient
                                evaluations from the `ropt` core during execution.
        """

    @abstractmethod
    def start(self, initial_values: NDArray[np.float64]) -> None:
        """Initiate the optimization process.

        This abstract method must be implemented by concrete `Backend`
        subclasses to start the optimization process. It takes the initial set
        of variable values as input.

        During execution, the implementation should use the
        [`OptimizerCallback`][ropt.core.OptimizerCallback] (provided during
        initialization) to request necessary function and gradient evaluations
        from the `ropt` core.

        Args:
            initial_values: A 1D NumPy array representing the starting variable
                            values for the optimization.
        """

    @property
    def allow_nan(self) -> bool:
        """Indicate whether the optimizer can handle NaN function values.

        If an optimizer algorithm can gracefully handle `NaN` (Not a Number)
        objective function values, its implementation should override this
        property to return `True`.

        This is particularly relevant in ensemble-based optimization where
        evaluations might fail for all realizations. When `allow_nan` is `True`,
        setting [`realization_min_success`][ropt.config.RealizationsConfig] to
        zero allows the evaluation process to return `NaN` instead of raising an
        error, enabling the optimizer to potentially continue.

        Returns:
            `True` if the optimizer supports NaN function values.
        """
        return False

    @property
    def is_parallel(self) -> bool:
        """Indicate whether the optimizer allows parallel evaluations.

        If an optimizer algorithm is designed to evaluate multiple variable
        vectors concurrently, its implementation should override this property
        to return `True`.

        This information can be used by `ropt` or other components to manage
        resources or handle parallel execution appropriately.

        Returns:
            `True` if the optimizer allows parallel evaluations.
        """
        return False

    @abstractmethod
    def validate_options(self) -> None:
        """Validate the optimizer-specific options.

        This method is intended to validate the `options` dictionary, passed
        upon creation of the optimizer via the
        [`BackendConfig`][ropt.config.BackendConfig] configuration object. It
        should check if the options contains valid keys and values for the
        specified optimization `method`.

        Subclasses should override this method to implement validation logic
        specific to the methods they support, potentially using schema
        validation tools like Pydantic.

        The raised exception must be a ValueError, or derive from a ValueError.

        Note:
            It is expected that the optimizer either receives a dictionary, or a
            list of options. This method should test if the type of the options
            is as expected, and raise a `ValueError` with an appropriate message
            if this is not the case.

        Warning: Method name with prefix
            The method string may be prefixed in the form "backend/method", take
            this into account when parsing the method name.

        Warning: Handling the default method
            The the method string may be set to "default", in which case it should
            be mapped to the correct default method of the backend.

        Raises:
            ValueError: If the provided options are invalid.
        """
