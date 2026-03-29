"""This module defines the base class for optimization plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


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
