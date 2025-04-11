"""This module defines base classes and protocols for optimization plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol

from ropt.plugins.base import Plugin

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ropt.config.enopt import EnOptConfig


class OptimizerCallback(Protocol):
    """Defines the call signature for the optimizer evaluation callback.

    [`Optimizer`][ropt.plugins.optimizer.base.Optimizer] instances, implemented
    by optimization plugins, are initialized with a callback function conforming
    to this protocol. The optimizer uses this callback to request function and
    gradient evaluations from the `ropt` core during the optimization process.
    """

    def __call__(
        self,
        variables: NDArray[np.float64],
        /,
        *,
        return_functions: bool,
        return_gradients: bool,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Request function and/or gradient evaluations from the `ropt` core.

        This method is called by the optimizer implementation to obtain
        objective function values, constraint values, and their gradients for
        one or more sets of variable values.

        The `variables` argument can be a 1D array (single vector) or a 2D array
        (matrix where each row is a variable vector). Parallel or batch-based
        optimizers might provide a matrix, while others typically provide a
        single vector.

        The `return_functions` and `return_gradients` flags control what is
        computed and returned. The results are always a tuple `(functions,
        gradients)`.

        - **Functions Array:** If `return_functions` is `True`, this array contains
          the objective and non-linear constraint values.
            If `variables` was a vector, it's a 1D array:

                [objective, constraint1, constraint2, ...]

            If `variables` was a matrix, it's a 2D array where each row corresponds
            to a row in the input `variables`, with the same structure:

                [
                  [obj_row1, con1_row1, ...],
                  [obj_row2, con2_row2, ...],
                 ...
                ]

        - **Gradients Array:** If `return_gradients` is `True`, this array contains
          the gradients of the objective and non-linear constraints. It's always
          a 2D array where rows correspond to the objective/constraints and columns
          correspond to the variables:

                [
                  [grad_obj_var1,  grad_obj_var2,  ...],
                  [grad_con1_var1, grad_con1_var2, ...],
                  ...
                ]

          Currently, gradient evaluation is only requested for a single variable
          vector at a time.

        If `return_functions` and/or `return_gradients` are `False`, the
        corresponding results are empty arrays.


        Args:
            variables:        A 1D or 2D array of variable values to evaluate.
            return_functions: If `True`, compute and return function/constraint values.
            return_gradients: If `True`, compute and return gradient values.

        Returns:
            A tuple containing two NumPy arrays: (functions_array, gradients_array).
        """


class Optimizer(ABC):
    """Abstract Base Class for Optimizer Implementations.

    This class defines the fundamental interface for all concrete optimizer
    implementations within the `ropt` framework. Optimizer plugins provide
    classes derived from `Optimizer` that encapsulate the logic of specific
    optimization algorithms.

    Instances of `Optimizer` subclasses are created by their corresponding
    [`OptimizerPlugin`][ropt.plugins.optimizer.base.OptimizerPlugin] factories.
    They are initialized with an
    [`EnOptConfig`][ropt.config.enopt.EnOptConfig] object detailing the
    optimization setup and an
    [`OptimizerCallback`][ropt.plugins.optimizer.base.OptimizerCallback] function.
    The callback is crucial as it allows the optimizer to request function and
    gradient evaluations from the `ropt` core during its execution.

    The optimization process itself is initiated by calling the `start` method,
    which must be implemented by subclasses.

    Subclasses must implement:
    - `__init__`: To accept the configuration and callback.
    - `start`: To contain the main optimization loop.

    Subclasses can optionally override:
    - `allow_nan`:   To indicate if the algorithm can handle NaN function values.
    - `is_parallel`: To indicate if the algorithm may perform parallel evaluations.
    """

    def __init__(  # noqa: B027
        self, config: EnOptConfig, optimizer_callback: OptimizerCallback
    ) -> None:
        """Initialize an optimizer object.

        The `config` object provides the desired configuration for the
        optimization process and should be used to set up the optimizer
        correctly before starting the optimization. The optimization will be
        initiated using the `start` method and will repeatedly require function
        and gradient evaluations at given variable vectors. The
        `optimizer_callback` argument provides the function that should be used
        to calculate the function and gradient values, which can then be
        forwarded to the optimizer.

        Args:
            config:             The optimizer configuration to used.
            optimizer_callback: The optimizer callback.
        """

    @abstractmethod
    def start(self, initial_values: NDArray[np.float64]) -> None:
        """Initiate the optimization process.

        This abstract method must be implemented by concrete `Optimizer`
        subclasses to start the optimization process. It takes the initial set
        of variable values as input.

        During execution, the implementation should use the
        [`OptimizerCallback`][ropt.plugins.optimizer.base.OptimizerCallback]
        (provided during initialization) to request necessary function and
        gradient evaluations from the `ropt` core.

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
        setting [`realization_min_success`][ropt.config.enopt.RealizationsConfig]
        to zero allows the evaluation process to return `NaN` instead of raising
        an error, enabling the optimizer to potentially continue.

        Returns:
            `True` if the optimizer supports NaN function values.
        """
        return False

    @property
    def is_parallel(self) -> bool:
        """Indicate whether the optimizer alows parallel evaluations.

        If an optimizer algorithm is designed to evaluate multiple variable
        vectors concurrently, its implementation should override this property
        to return `True`.

        This information can be used by `ropt` or other components to manage
        resources or handle parallel execution appropriately.

        Returns:
            `True` if the optimizer allows parallel evaluations.
        """
        return False


class OptimizerPlugin(Plugin):
    """Abstract Base Class for Optimizer Plugins (Factories).

    This class defines the interface for plugins responsible for creating
    [`Optimizer`][ropt.plugins.optimizer.base.Optimizer] instances. These plugins
    act as factories for specific optimization algorithms or backends.

    During plan execution, the [`PluginManager`][ropt.plugins.PluginManager]
    identifies the appropriate optimizer plugin based on the configuration and
    uses its `create` class method to instantiate the actual `Optimizer` object
    that will perform the optimization.
    """

    @classmethod
    @abstractmethod
    def create(
        cls, config: EnOptConfig, optimizer_callback: OptimizerCallback
    ) -> Optimizer:
        """Create an Optimizer instance.

        This abstract class method serves as a factory for creating concrete
        [`Optimizer`][ropt.plugins.optimizer.base.Optimizer] objects. Plugin
        implementations must override this method to return an instance of their
        specific `Optimizer` subclass.

        The [`PluginManager`][ropt.plugins.PluginManager] calls this method when
        an optimization step requires an optimizer provided by this plugin.

        Args:
            config:             The  configuration object containing the
                                optimization settings.
            optimizer_callback: The callback function used by the optimizer to
                                request evaluations.

        Returns:
            An initialized instance of an `Optimizer` subclass.
        """

    @classmethod
    def validate_options(
        cls,
        method: str,
        options: dict[str, Any] | list[str] | None,
    ) -> None:
        """Validate the optimizer-specific options for a given method.

        This class method is intended to check if the `options` dictionary,
        typically provided in the
        [`OptimizerConfig`][ropt.config.enopt.OptimizerConfig], contains valid
        keys and values for the specified optimization `method` supported by
        this plugin.

        This default implementation performs no validation. Subclasses should
        override this method to implement validation logic specific to the
        methods they support, potentially using schema validation tools like
        Pydantic.

        Args:
            method:  The specific optimization method name (e.g., "slsqp",
                     "my_optimizer/variant1").
            options: The dictionary or a list of strings of options.

        Raises:
            Exception: If the provided options are invalid for the specified
                       method.
        """
