"""This module defines a basic variable scaling transform."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ropt.config import (
    NonlinearConstraintTransformConfig,
    ObjectiveTransformConfig,
    VariableTransformConfig,
)

from .base import NonlinearConstraintTransform, ObjectiveTransform, VariableTransform

DEFAULT_VARIABLE_TRANSFORM_METHODS = {"scaler"}
DEFAULT_OBJECTIVE_TRANSFORM_METHODS = {"scaler"}
DEFAULT_NONLINEAR_CONSTRAINT_TRANSFORM_METHODS = {"scaler"}


class DefaultVariableTransform(VariableTransform):
    r"""Linearly scales and shifts variables between domains.

    This class implements a linear transformation for variables, allowing
    for scaling and shifting between the user-defined domain and the
    optimizer's internal domain. The transformation is defined by a scaling
    factor and an offset for each variable.

    The transformation from the user domain to the optimizer domain is given by:

    $$x_{opt} = \frac{(x_{\textrm{user}} - \textrm{offset})}{\textrm{scale}}$$

    The transformation from the optimizer domain back to the user domain is:

    $$x_{user} = x_{\textrm{opt}} * {\textrm{scale}} + {\textrm{offset}}$$

    This transformation can be used to improve the performance of the
    optimizer by working with variables that are scaled to a more suitable
    range or centered around a specific value.
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
        self._scales: NDArray[np.float64] | None = scales
        self._offsets: NDArray[np.float64] | None = offsets
        self._equation_scaling: NDArray[np.float64] | None = None

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

    def init(self, mask: NDArray[np.bool_]) -> None:
        """Apply mask: set scales to 1 and offsets to 0 for unmasked variables.

        Args:
            mask: Boolean array (`True` = this transform applies).
        """
        if self._scales is not None:
            self._scales = np.where(mask, self._scales, 1.0)
        if self._offsets is not None:
            self._offsets = np.where(mask, self._offsets, 0.0)


class DefaultObjectiveTransform(ObjectiveTransform):
    r"""Linearly scales objectives between domains.

    Divides by `scales` when going to the optimizer domain, multiplies when
    returning to the user domain.
    """

    def __init__(
        self,
        transform_config: ObjectiveTransformConfig,
    ) -> None:
        """Initialize the objective scaler.

        Reads `scales` from the transform configuration options.

        Args:
            transform_config: The transform configuration.
        """
        self._scales: NDArray[np.float64] | None = transform_config.options.get(
            "scales", None
        )
        self._mask: NDArray[np.bool_] | None = None

    def to_optimizer(self, objectives: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply `objectives / scales`.

        Args:
            objectives: Objective values in the user domain.

        Returns:
            Transformed objectives in the optimizer domain.
        """
        if self._scales is not None:
            return objectives / self._scales
        return objectives

    def from_optimizer(self, objectives: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply `objectives * scales`.

        Args:
            objectives: Objective values in the optimizer domain.

        Returns:
            Transformed objectives in the user domain.
        """
        if self._scales is not None:
            return objectives * self._scales
        return objectives

    def update(self, scales: ArrayLike) -> None:
        """Set new scaling factors (applies mask if previously initialized).

        Args:
            scales: The new scaling factors.
        """
        self._scales = np.asarray(scales, dtype=np.float64)
        if self._mask is not None:
            self._scales = np.where(self._mask, self._scales, 1.0)

    def init(self, mask: NDArray[np.bool_]) -> None:
        """Apply mask: set scales to 1 for unmasked objectives.

        Args:
            mask: Boolean array (`True` = this transform applies).
        """
        if self._scales is not None:
            self._scales = np.where(mask, self._scales, 1.0)
        self._mask = mask


class DefaultNonlinearConstraintTransform(NonlinearConstraintTransform):
    r"""Linearly scales constraints between domains.

    Divides by `scales` when going to the optimizer domain, multiplies when
    returning to the user domain. Also scales RHS bounds consistently.
    """

    def __init__(
        self,
        transform_config: NonlinearConstraintTransformConfig,
    ) -> None:
        """Initialize the constraint scaler.

        Reads `scales` from the transform configuration options.

        Args:
            transform_config: The transform configuration.
        """
        self._scales: NDArray[np.float64] | None = transform_config.options.get(
            "scales", None
        )
        self._mask: NDArray[np.bool_] | None = None

    def to_optimizer(self, constraints: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply `constraints / scales`.

        Args:
            constraints: Constraint values in the user domain.

        Returns:
            Transformed constraint values in the optimizer domain.
        """
        if self._scales is not None:
            return constraints / self._scales
        return constraints

    def from_optimizer(self, constraints: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply `constraints * scales`.

        Args:
            constraints: Constraint values in the optimizer domain.

        Returns:
            Transformed constraint values in the user domain.
        """
        if self._scales is not None:
            return constraints * self._scales
        return constraints

    def bounds_to_optimizer(
        self, lower_bounds: NDArray[np.float64], upper_bounds: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Apply `bounds / scales`.

        Args:
            lower_bounds: Lower RHS bounds in user domain.
            upper_bounds: Upper RHS bounds in user domain.

        Returns:
            Tuple of (lower_bounds, upper_bounds) in optimizer domain.
        """
        if self._scales is not None:
            return lower_bounds / self._scales, upper_bounds / self._scales
        return lower_bounds, upper_bounds

    def nonlinear_constraint_diffs_from_optimizer(
        self, lower_diffs: NDArray[np.float64], upper_diffs: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Apply `diffs * scales`.

        Args:
            lower_diffs: Constraint value minus lower bound (optimizer domain).
            upper_diffs: Constraint value minus upper bound (optimizer domain).

        Returns:
            Tuple of (lower_diffs, upper_diffs) in user domain.
        """
        if self._scales is not None:
            return lower_diffs * self._scales, upper_diffs * self._scales
        return lower_diffs, upper_diffs

    def update(self, scales: ArrayLike) -> None:
        """Set new scaling factors (applies mask if previously initialized).

        Args:
            scales: The new scaling factors.
        """
        self._scales = np.asarray(scales, dtype=np.float64)
        if self._mask is not None:
            self._scales = np.where(self._mask, self._scales, 1.0)

    def init(self, mask: NDArray[np.bool_]) -> None:
        """Apply mask: set scales to 1 for unmasked constraints.

        Args:
            mask: Boolean array (`True` = this transform applies).
        """
        if self._scales is not None:
            self._scales = np.where(mask, self._scales, 1.0)
        self._mask = mask
