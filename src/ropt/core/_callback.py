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

    Bundles the objective function values, gradient values, and any updated
    non-linear constraint bounds returned by an
    [`OptimizerCallback`][ropt.core.OptimizerCallback] evaluation. `functions`
    and `gradients` are `None` unless requested and successfully computed;
    `nonlinear_constraint_bounds` is `None` if the run has no non-linear
    constraints, and otherwise holds their bounds in the optimizer domain. The
    bounds are reported after every evaluation because auto-scaling can change
    them once, when the scales are estimated from the first batch.

    `functions` and `gradients` follow a fixed shape:

    - **Functions array:** the objective and non-linear constraint values. A
        vector `variables` gives a 1D array `[objective, constraint1, ...]`; a
        matrix `variables` gives a 2D array with one such row per input row.
    - **Gradients array:** always 2D, with one row per objective/constraint and
        one column per variable:

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

    Optimizers use this callback to request function and gradient evaluations
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

        `variables` is a 1D array for a single vector, or a 2D array (one
        vector per row) for optimizers that evaluate a batch at once.
        `return_functions` and `return_gradients` select what the returned
        [`OptimizerCallbackResult`][ropt.core.OptimizerCallbackResult] computes.

        Args:
            variables:        A 1D or 2D array of variable values to evaluate.
            return_functions: If `True`, compute and return function/constraint values.
            return_gradients: If `True`, compute and return gradient values.

        Returns:
            A data structure with the results.
        """
