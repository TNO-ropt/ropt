"""This module defines a basic variable scaling transform."""

import numpy as np
from numpy.typing import NDArray

from ropt.config import (
    VariableTransformConfig,
)
from ropt.plugins.transforms import (
    VariableTransformPlugin,
)

from .base import VariableTransform

DEFAULT_VARIABLE_TRANSFORM_METHODS = {"scaler"}
DEFAULT_OBJECTIVE_TRANSFORM_METHODS = {"scaler"}
DEFAULT_NONLINEAR_CONSTRAINT_TRANSFORM_METHODS = {"scaler"}


def _check_mask_size(mask: NDArray[np.bool_], size: int, what: str) -> None:
    # A mismatched mask would otherwise broadcast silently or raise a bare numpy error.
    if mask.size != size:
        msg = f"transform mask size ({mask.size}) does not match {what} ({size})"
        raise ValueError(msg)


class DefaultVariableTransform(VariableTransform):
    """Linearly scales and shifts variables between domains.

    See [Transforms](../optimizer_setup/variable_transforms.md#defaultvariabletransform)
    for the formulas and configuration options.
    """

    def __init__(
        self,
        transform_config: VariableTransformConfig,
    ) -> None:
        """Initialize the variable scaler.

        Reads `scales` and `offsets` from the transform configuration options.
        If both are provided, they are broadcasted to the same length.

        Args:
            transform_config: The transform configuration.
        """
        scales = transform_config.options.get("scales", None)
        offsets = transform_config.options.get("offsets", None)
        if scales is not None and offsets is not None:
            scales, offsets = np.broadcast_arrays(scales, offsets)
        self._scales: NDArray[np.float64] | None = (
            None if scales is None else np.asarray(scales, dtype=np.float64)
        )
        self._offsets: NDArray[np.float64] | None = (
            None if offsets is None else np.asarray(offsets, dtype=np.float64)
        )
        self._equation_scaling: NDArray[np.float64] | None = None
        self._mask: NDArray[np.bool_] | None = transform_config.mask
        if self._mask is not None:
            if self._scales is not None:
                _check_mask_size(self._mask, self._scales.size, "scales")
            if self._offsets is not None:
                _check_mask_size(self._mask, self._offsets.size, "offsets")

    def to_optimizer(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply `(values - offset) / scale`.

        Args:
            values: Variable values in the user domain.

        Returns:
            Transformed values in the optimizer domain.
        """
        if self._offsets is not None:
            values = values.copy() - self._offsets
        if self._scales is not None:
            values = values.copy() / self._scales
        return values

    def from_optimizer(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply `values * scale + offset`.

        Args:
            values: Variable values in the optimizer domain.

        Returns:
            Transformed values in the user domain.
        """
        if self._scales is not None:
            values = values.copy() * self._scales
        if self._offsets is not None:
            values = values.copy() + self._offsets
        return values

    def magnitudes_to_optimizer(
        self, values: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Apply `values / scale`.

        Args:
            values: Perturbation magnitudes in the user domain.

        Returns:
            Magnitudes in the optimizer domain.
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
        r"""Transform linear constraints to the optimizer domain.

        The set of linear constraints can be represented by a matrix equation:
        $\mathbf{A} \mathbf{x} = \mathbf{b}$.

        If the linear transformation of variables to the optimizer domain is:

        $$ \hat{\mathbf{x}} = \mathbf{S} \mathbf{x} + \mathbf{o}$$

        then the coefficients and right-hand-side values become:

        $$ \begin{align}
            \hat{\mathbf{A}} &= \mathbf{A} \mathbf{S}^{-1} \\ \hat{\mathbf{b}}
            &= \mathbf{b} + \mathbf{A}\mathbf{S}^{-1}\mathbf{o}
        \end{align}$$

        where $S$ is a diagonal matrix with scaling factors and $o$ are offsets.

        The resulting equations are further scaled by dividing by the maximum
        absolute coefficient in each equation.

        Args:
            coefficients: The coefficient matrix of the linear constraints.
            lower_bounds: The lower bounds on the right-hand-side values.
            upper_bounds: The upper bounds on the right-hand-side values.

        Returns:
            A tuple containing the transformed coefficient matrix and bounds.
        """
        if self._offsets is not None:
            offsets = np.matmul(coefficients, self._offsets)
            lower_bounds = lower_bounds.copy() - offsets
            upper_bounds = upper_bounds.copy() - offsets
        if self._scales is not None:
            coefficients = coefficients.copy() * self._scales
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
        """Scale differences back by `* scale`.

        Args:
            lower_diffs: Variable value minus lower bound (optimizer domain).
            upper_diffs: Variable value minus upper bound (optimizer domain).

        Returns:
            A tuple of (lower_diffs, upper_diffs) in user domain.
        """
        if self._scales is not None:
            lower_diffs = lower_diffs.copy() * self._scales
            upper_diffs = upper_diffs.copy() * self._scales
        return lower_diffs, upper_diffs

    def linear_constraints_diffs_from_optimizer(
        self, lower_diffs: NDArray[np.float64], upper_diffs: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Re-scale by the equation weights stored during `linear_constraints_to_optimizer`.

        Args:
            lower_diffs: Linear constraint value minus lower bound.
            upper_diffs: Linear constraint value minus upper bound.

        Returns:
            A tuple of (lower_diffs, upper_diffs) in user domain.
        """
        if self._equation_scaling is not None:
            lower_diffs = lower_diffs.copy() * self._equation_scaling
            upper_diffs = upper_diffs.copy() * self._equation_scaling
        return lower_diffs, upper_diffs

    def set_free_mask(self, mask: NDArray[np.bool_]) -> None:
        """Neutralize scales and offsets outside the configured mask and free variables.

        Args:
            mask: Boolean array (`True` = the variable is free).
        """
        if self._mask is not None:
            _check_mask_size(self._mask, mask.size, "the number of variables")
            mask = np.logical_and(mask, self._mask)
        if self._scales is not None:
            self._scales = np.where(mask, self._scales, 1.0)
        if self._offsets is not None:
            self._offsets = np.where(mask, self._offsets, 0.0)


class DefaultVariableTransformPlugin(VariableTransformPlugin):
    """Default variable transform plugin class."""

    @classmethod
    def create(
        cls,
        config: VariableTransformConfig,
    ) -> DefaultVariableTransform:
        """Create a DefaultVariableTransform instance.

        Args:
            config: The variable transform configuration.

        Returns:
            A new `DefaultVariableTransform`.
        """
        return DefaultVariableTransform(config)

    @classmethod
    def is_supported(cls, method: str) -> bool:  # ruff: ignore[undocumented-public-method]
        return method.lower() in (DEFAULT_VARIABLE_TRANSFORM_METHODS | {"default"})
