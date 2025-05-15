"""This module defines classes and protocols for optimization callbacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@dataclass
class OptimizerCallbackResult:
    """Holds the results from an optimizer callback evaluation.

    This dataclass is used to structure the output of the
    [`OptimizerCallback`][ropt.plugins.optimizer.base.OptimizerCallback].
    It bundles the objective function values, gradient values, and any
    updated non-linear constraint bounds that result from an evaluation
    request.

    The `functions` attribute will contain a NumPy array of the objective
    function value(s) if they were requested and successfully computed,
    otherwise it will be `None`. Similarly, the `gradients` attribute will
    hold a NumPy array of gradient values if requested and computed, and
    `None` otherwise.

    The `nonlinear_constraint_bounds` attribute is a tuple containing two
    NumPy arrays: the first for lower bounds and the second for upper bounds
    of any non-linear constraints. This will be `None` if there are no
    non-linear constraints or if their bounds were not updated during the
    callback.

    The `functions` and `gradients` fields must be structured as follows:

    - **Functions Array:** This array contains the objective and non-linear
        constraint values. If `variables` was a vector, it's a 1D array:

            [objective, constraint1, constraint2, ...]

        If `variables` was a matrix, it's a 2D array where each row corresponds
        to a row in the input `variables`, with the same structure:

            [
                [obj_row1, con1_row1, ...],
                [obj_row2, con2_row2, ...],
                ...
            ]

    - **Gradients Array:** This array contains the gradients of the objective
        and non-linear constraints. It's always a 2D array where rows correspond
        to the objective/constraints and columns correspond to the variables:

            [
                [grad_obj_var1,  grad_obj_var2,  ...],
                [grad_con1_var1, grad_con1_var2, ...],
                ...
            ]

    Attributes:
        functions: Objective function value(s).
        gradients: Gradient values.
        nonlinear_constraint_bounds: Updated non-linear constraint lower and upper bounds.
    """

    functions: NDArray[np.float64] | None
    gradients: NDArray[np.float64] | None
    nonlinear_constraint_bounds: tuple[NDArray[np.float64], NDArray[np.float64]] | None


class OptimizerCallback(Protocol):
    """Defines the call signature for the optimizer evaluation callback.

    Optimizers uses this callback to request function and gradient evaluations
    from the `ropt` core during the optimization process.
    """

    def __call__(
        self,
        variables: NDArray[np.float64],
        /,
        *,
        return_functions: bool,
        return_gradients: bool,
    ) -> OptimizerCallbackResult:
        """Request function and/or gradient evaluations from the `ropt` core.

        This method is called by the optimizer implementation to obtain
        objective function values, constraint values, and their gradients for
        one or more sets of variable values. In addition other update
        information, such as non-linear constraint bounds may be returned.

        The `variables` argument can be a 1D array (single vector) or a 2D array
        (matrix where each row is a variable vector). Parallel or batch-based
        optimizers might provide a matrix, while others typically provide a
        single vector.

        The `return_functions` and `return_gradients` flags control what is
        computed and returned in a
        [`OptimizerCallbackResult`][ropt.optimization.OptimizerCallbackResult]
        structure.

        Args:
            variables:        A 1D or 2D array of variable values to evaluate.
            return_functions: If `True`, compute and return function/constraint values.
            return_gradients: If `True`, compute and return gradient values.

        Returns:
            A data structure with the results.
        """
