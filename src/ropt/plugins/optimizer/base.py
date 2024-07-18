"""This module defines base classes and protocols for optimization plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, Tuple

from ropt.plugins.base import Plugin

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ropt.config.enopt import EnOptConfig


class OptimizerCallback(Protocol):
    """Protocol for the optimizer callback.

    Optimization plugins implement optimizer classes derived from the
    [`Optimizer`][ropt.plugins.optimizer.base.Optimizer] abstract base class.
    Objects of these classes are initialized with a callback function that
    follows the call signature defined here. This callback should be used to
    request the function and gradient evaluations that the optimizer needs.
    """

    def __call__(
        self,
        variables: NDArray[np.float64],
        /,
        *,
        return_functions: bool,
        return_gradients: bool,
        allow_nan: bool = False,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The signature of the optimizer callback.

        The optimizer callback expects a vector or matrix with variables to
        evaluate. Discrete optimizers may request function evaluations for
        multiple variable vectors, passed as the rows of a matrix.
        Gradient-based methods may currently only pass a single variable vector
        at a time.

        The `return_functions` and `return_gradients` flags determine whether
        functions and/or gradients are to be evaluated. The results are returned
        as a tuple of arrays, one for functions and constraints, the other for
        gradients. If one of `return_functions` or `return_gradients` is
        `False`, the corresponding result is an empty array.

        Multiple function evaluations are returned as a matrix where the rows
        are the result vectors for each evaluation. The first element of a
        result vector is the value of the objective value, and the remaining
        elements are the values of the non-linear constraints.

        The gradients of the objective function and the non-linear constraints
        are returned as a matrix. The first row contains the gradient of the
        objective function, while the remaining rows contain the gradients of
        the non-linear constraints. Gradient-based methods currently support
        only a single evaluation, hence there is also only a single result.

        In most cases, the optimizer cannot handle failed function evaluations,
        which are indicated by `NaN` values. Some optimizers, in particular
        those that use multiple function evaluations do determine a next step
        are robust in this regard. By returning `allow_nan=True`, these
        optimizers can indicate that this is the case.

        Args:
            variables:        The variable vector or matrix to evaluate
            return_functions: If `True`, evaluate and return functions
            return_gradients: If `True`, evaluate and return gradients
            allow_nan:        If `True`, accept `NaN` values

        Returns:
            A tuple with function and gradient values.
        """


class Optimizer(ABC):
    """Abstract base for optimizer classes.

    Optimizers provided by optimizer plugins should derive from the `Optimizer`
    abstract base class, which specifies the requirements for the class
    constructor (`__init__`) and also includes a `start` method used to initiate
    the optimization process.
    """

    @abstractmethod
    def __init__(
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
            config:             The optimizer configuration to used
            optimizer_callback: The optimizer callback
        """

    @abstractmethod
    def start(self, initial_values: NDArray[np.float64]) -> None:
        """Start the optimization.

        This method must be implemented to run the optimization using the
        provided initial values, collecting function and gradient evaluations as
        needed by calling the callback passed via the `optimizer_callback`
        argument during initialization.

        Args:
            initial_values: Vector of variables to start the optimization with.
        """

    @property
    @abstractmethod
    def allow_nan(self) -> bool:
        """Return `True` if a `NaN` is a valid function value.

        If the optimizer can handle `NaN` function values, it may implement this
        method to return `True`. This enables handling cases where all function
        evaluations in an ensemble fail. By setting the
        [`realization_min_success`][ropt.config.enopt.RealizationsConfig]
        configuration parameter to zero, the ensemble optimization plan can then
        be allowed to return a `NaN` value instead of raising an error.

        Returns:
            `True` if `NaN` is allowed.
        """


class OptimizerPlugin(Plugin):
    """Abstract base class for optimizer plugins.

    Optimizers are implemented through optimizer plugins that must derive from
    this base class. Plugins can be built-in, installed via a plugin mechanism,
    or loaded dynamically. During execution of an optimization plan, the
    required optimizer plugin will be located via the
    [`PluginManager`][ropt.plugins.PluginManager] and used to create
    [`Optimizer`][ropt.plugins.optimizer.base.Optimizer] objects via its
    `create` function.
    """

    @abstractmethod
    def create(
        self, config: EnOptConfig, optimizer_callback: OptimizerCallback
    ) -> Optimizer:
        """Create an optimizer.

        This is a factory function for instantiating optimizer objects
        implemented in the plugin.

        Args:
            config:             The optimizer configuration to used
            optimizer_callback: The optimizer callback
        """
