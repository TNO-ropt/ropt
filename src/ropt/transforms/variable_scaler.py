"""This module defines a basic variable scaling transform."""

import numpy as np
from numpy.typing import NDArray

from .base import VariableTransform


class VariableScaler(VariableTransform):
    """Transform class for scaling variables."""

    def __init__(
        self, scales: NDArray[np.float64] | None, offsets: NDArray[np.float64] | None
    ) -> None:
        """Initialize the scaler.

        This scaler applies a linear scaling to the variables, defined
        by a scaling vector and an offset vector.

        if `scales` and `offsets` are both not `None`, they are broadcasted
        to the same length.

        Args:
            scales:  The scaling factors.
            offsets: The offset values.
        """
        if scales is not None and offsets is not None:
            scales, offsets = np.broadcast_arrays(scales, offsets)
        self._scales = scales
        self._offsets = offsets
        self._equation_scaling: NDArray[np.float64] | None = None

    def to_optimizer(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Implement the scaling to optimizer space.

        The values may consist of an array with multiple dimensions. It is
        assumed that the last axis contains the variable values. Should this
        method be used in a context where that is not the case, adjust the
        order of the axis accordingly before and after calling this method.

        Args:
            values: The values to be scaled.

        Returns:
            The scaled values.
        """
        if self._offsets is not None:
            values = values - self._offsets
        if self._scales is not None:
            values = values / self._scales
        return values

    def from_optimizer(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Implement the scaling from optimizer space.

        The values may consist of an array with multiple dimensions. It is
        assumed that the last axis contains the variable values. Should this
        method be used in a context where that is not the case, adjust the
        order of the axis accordingly before and after calling this method.

        Args:
            values: The values to be scaled.

        Returns:
            The scaled values.
        """
        if self._scales is not None:
            values = values * self._scales
        if self._offsets is not None:
            values = values + self._offsets
        return values

    def magnitudes_to_optimizer(
        self, values: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Implement the transformation of perturbation magnitudes.

        Args:
            values: The values to be transformed.

        Returns:
            The transformed values.
        """
        if self._scales is not None:
            return values / self._scales
        return values

    def linear_constraints_to_optimizer(
        self,
        coefficients: NDArray[np.float64],
        lower_bounds: NDArray[np.float64],
        upper_bounds: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Implement the transformation of linear constraints.

        Args:
            coefficients: The coefficient matrix of the linear constraints.
            lower_bounds: The lower bounds on the right-hand-sides of the constraint equations.
            upper_bounds: The upper bounds on the right-hand-sides of the constraint equations.

        Returns:
            The transformed coefficient matrix and right-hand side bounds.
        """
        if self._offsets is not None:
            offsets = np.matmul(coefficients, self._offsets)
            lower_bounds = lower_bounds - offsets
            upper_bounds = upper_bounds - offsets
        if self._scales is not None:
            coefficients = coefficients * self._scales
        self._equation_scaling = np.max(np.abs(coefficients), axis=-1)
        assert self._equation_scaling is not None
        return (
            coefficients / self._equation_scaling[:, np.newaxis],
            lower_bounds / self._equation_scaling,
            upper_bounds / self._equation_scaling,
        )

    def bound_constraint_diffs_from_optimizer(
        self, lower_diffs: NDArray[np.float64], upper_diffs: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Implement the transformation of bound constraint diffs from optimizer space.

        Args:
            lower_diffs: The lower differences to be transformed.
            upper_diffs: The upper differences to be transformed.

        Returns:
            The transformed values.
        """
        if self._scales is not None:
            lower_diffs = lower_diffs * self._scales
            upper_diffs = upper_diffs * self._scales
        return lower_diffs, upper_diffs

    def linear_constraints_diffs_from_optimizer(
        self, lower_diffs: NDArray[np.float64], upper_diffs: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Implement the transformation of bound constraint diffs from optimizer space.

        Args:
            lower_diffs: The lower differences to be transformed.
            upper_diffs: The upper differences to be transformed.

        Returns:
            The transformed values.
        """
        if self._equation_scaling is not None:
            lower_diffs = lower_diffs * self._equation_scaling
            upper_diffs = upper_diffs * self._equation_scaling
        return lower_diffs, upper_diffs
