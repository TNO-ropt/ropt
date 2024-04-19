"""This module defines the protocol to be followed by optimization backends.

Optimization backends can be added via the plugin mechanism to implement
additional optimization algorithms. Any object that follows the
[`Optimizer`][ropt.plugins.optimizer.protocol.Optimizer]
protocol may be installed as a plugin.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Tuple

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ropt.config.enopt import EnOptConfig


class OptimizerCallback(Protocol):
    """Protocol for the optimizer callback.

    Optimization plugins implement optimizer classes according to the
    [`Optimizer`][ropt.plugins.optimizer.protocol.Optimizer] protocol. Objects
    of these classes are initialized with a callback function that follows the
    call signature defined here. This callback should be used to request the
    function and gradient evaluations that the optimizer needs.
    """

    def __call__(
        self,
        variables: NDArray[np.float64],
        /,
        *,
        return_functions: bool,
        return_gradients: bool,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The signature of the optimizer callback.

        The optimizer callback expects a vector or matrix with variables to
        evaluate. Discrete optimizers may request function evaluations for
        multiple variable vectors, passed as the rows of a matrix.
        Gradient-based algorithms may currently only pass a single variable
        vector at a time.

        The `return_functions` and `return_gradients` flags determine whether
        functions and/or gradients are to be evaluated. The results are returned
        as a tuple of arrays, one for functions and constraints, the other for
        gradients. If one of `return_functions` or `return_gradients` is `False`,
        the corresponding result is an empty array.

        Multiple function evaluations are returned as a matrix where the rows
        are the result vectors for each evaluation. The first element of a
        result vector is the value of the objective value, and the remaining
        elements are the values of the non-linear constraints.

        The gradients of the objective function and the non-linear constraints
        are returned as a matrix. The first row contains the gradient of the
        objective function, while the remaining rows contain the gradients of
        the non-linear constraints. Gradient-based methods currently support
        only a single evaluation, hence there is also only a single result.

        Args:
            variables:        The variable vector or matrix to evaluate.
            return_functions: If `True`, evaluate and return functions.
            return_gradients: If `True`, evaluate and return gradients.

        Returns:
            A tuple with function and gradient values.
        """


class Optimizer(Protocol):
    """Protocol for optimizer classes.

    `ropt` employs plugins to implement optimizers that are called during an
    optimization workflow. Optimizers should adhere to the `Optimizer` protocol,
    which specifies the requirements for the class constructor (`__init__`) and
    also includes a `start` method used to initiate the optimization process.
    """

    def __init__(
        self, config: EnOptConfig, optimizer_callback: OptimizerCallback
    ) -> None:
        """Initialize an optimizer object.

        The `config` object defines how the optimization run should be
        configured, while the `optimizer_callback` should be used to evaluate
        functions and gradients that are needed during optimization.

        Args:
            config:             The optimizer configuration to used
            optimizer_callback: The optimizer callback
        """

    def start(self, initial_values: NDArray[np.float64]) -> None:
        """Start the optimization.

        This method must be implemented to run the optimization, collecting
        functions and gradients as needed by calling the callback passed upon
        initialization.

        Args:
            initial_values: Vector of variables to start the optimization with.
        """
