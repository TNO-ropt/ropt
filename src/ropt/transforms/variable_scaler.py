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

    def forward(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Implement the forward scaling.

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

    def backward(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Implement the backward scaling.

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

    def transform_magnitudes(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Implement the forward transformation of perturbation magnitudes.

        Args:
            values: The values to be transformed.

        Returns:
            The transformed values.
        """
        if self._scales is not None:
            return values / self._scales
        return values

    def transform_linear_constraints(
        self,
        coefficients: NDArray[np.float64],
        lower_bounds: NDArray[np.float64],
        upper_bounds: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Implement the forward transformation of linear constraints.

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
        equation_scaling = np.max(np.abs(coefficients), axis=-1)
        return (
            coefficients / equation_scaling[:, np.newaxis],
            lower_bounds / equation_scaling,
            upper_bounds / equation_scaling,
        )
