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

    def forward(
        self,
        values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
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
        self, coefficients: NDArray[np.float64], rhs_values: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Implement the forward scaling of linear constraints.

        Args:
            coefficients: The coefficient matrix of the linear constraints.
            rhs_values:   The right-hand side values of the linear constraints.

        Returns:
            The scaled coefficients and right-hand side values.
        """
        if self._offsets is not None:
            rhs_values = rhs_values - np.matmul(coefficients, self._offsets)
        if self._scales is not None:
            coefficients = coefficients * self._scales
        return coefficients, rhs_values
