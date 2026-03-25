"""This module defines a basic variable scaling transform."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ropt.config import (
    NonlinearConstraintTransformConfig,
    ObjectiveTransformConfig,
    VariableTransformConfig,
)

from .base import NonLinearConstraintTransform, ObjectiveTransform, VariableTransform

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

        This scaler applies a linear transformation to variables, defined by
        scaling factors and offset values, defined in the transform configuration.

        If both `scales` and `offsets` are provided, they are broadcasted to
        ensure they have the same length.

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
        """Transform variable values to the optimizer domain.

        This method applies the linear scaling and offset transformation to
        variable values, mapping them from the user-defined domain to the
        optimizer's internal domain.

        The transformation is defined as: `x_opt = (x_user - offset) / scale`.

        The input `values` may be a multi-dimensional array. It is assumed that
        the last axis of the array represents the variable values. If this is
        not the case, you must adjust the order of the axes before and after
        calling this method.

        Args:
            values: The variable values in the user domain to be transformed.

        Returns:
            The transformed variable values in the optimizer domain.
        """
        if self._offsets is not None:
            values = values.copy() - self._offsets
        if self._scales is not None:
            values = values.copy() / self._scales
        return values

    def from_optimizer(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform variable values to the user domain.

        This method applies the inverse linear scaling and offset transformation
        to variable values, mapping them from the optimizer's internal domain
        back to the user-defined domain.

        The transformation is defined as: `x_user = x_opt * scale + offset`.

        The input `values` may be a multi-dimensional array. It is assumed that
        the last axis of the array represents the variable values. If this is
        not the case, you must adjust the order of the axes before and after
        calling this method.

        Args:
            values: The variable values in the optimizer domain to be
                transformed.

        Returns:
            The transformed variable values in the user domain.
        """
        if self._scales is not None:
            values = values.copy() * self._scales
        if self._offsets is not None:
            values = values.copy() + self._offsets
        return values

    def magnitudes_to_optimizer(
        self, values: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Transform perturbation magnitudes to the optimizer domain.

        This method transforms perturbation magnitudes, typically used in
        stochastic gradient-based algorithms, from the user-defined domain to
        the optimizer's internal domain. The transformation ensures that the
        perturbations are applied correctly in the optimizer's space, which may
        have different scaling or units than the user domain.

        The transformation is defined as: `x_opt = x_user / scale`.

        Args:
            values: The perturbation magnitudes in the user domain.

        Returns:
            The transformed perturbation magnitudes in the optimizer domain.
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

        This method transforms linear constraints, defined by their coefficients
        and right-hand-side bounds, from the user-defined domain to the
        optimizer's internal domain. This transformation accounts for the
        scaling and shifting applied to the variables and ensures that the
        constraints remain valid in the optimizer's space.

        The set of linear constraints can be represented by a matrix equation:
        $\mathbf{A} \mathbf{x} = \mathbf{b}$.

        When linearly transforming variables to the optimizer domain, the
        coefficients ($\mathbf{A}$) and right-hand-side values ($\mathbf{b}$)
        must be converted to remain valid (see also the configuration for
        [linear constraints][ropt.config.LinearConstraintsConfig]). If the
        linear transformation of the variables to the optimizer domain is given
        by:

        $$ \hat{\mathbf{x}} = \mathbf{S} \mathbf{x} + \mathbf{o}$$

        then the coefficients and right-hand-side values must be transformed as
        follows:

        $$ \begin{align}
            \hat{\mathbf{A}} &= \mathbf{A} \mathbf{S}^{-1} \\ \hat{\mathbf{b}}
            &= \mathbf{b} + \mathbf{A}\mathbf{S}^{-1}\mathbf{o}
        \end{align}$$

        where $S$ is a diagonal matrix with scaling factors on the diagonal and
        $o$ are the offsets.

        The resulting equations are further scaled by dividing them by maximum
        of the absolute values of the coefficients in each equation.

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
        """Transform bound constraint differences to the user domain.

        This method transforms the differences between variable values and their
        lower/upper bounds from the optimizer's internal domain back to the
        user-defined domain. These differences are used to report constraint
        violations.

        For example, if variables are scaled in the optimizer domain, the
        differences between the variables and their bounds must be scaled back
        to the user domain to accurately reflect the constraint violations in
        the user's original units.

        The transformation is defined as: `x_user = x_opt * scale`.

        Args:
            lower_diffs: The differences between the variable values and their
                lower bounds.
            upper_diffs: The differences between the variable values and their
                upper bounds.

        Returns:
            A tuple containing the transformed lower and upper differences.
        """
        if self._scales is not None:
            lower_diffs = lower_diffs.copy() * self._scales
            upper_diffs = upper_diffs.copy() * self._scales
        return lower_diffs, upper_diffs

    def linear_constraints_diffs_from_optimizer(
        self, lower_diffs: NDArray[np.float64], upper_diffs: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Transform linear constraint differences to the user domain.

        This method transforms the differences between linear constraint values
        and their lower/upper bounds from the optimizer's internal domain back
        to the user-defined domain. These differences are used to report
        constraint violations.

        This is implemented by re-scaling the equations with the weights that
        were determined and stored by the
        [`linear_constraints_to_optimizer`][ropt.transforms.VariableTransform.linear_constraints_to_optimizer]
        method.

        Args:
            lower_diffs: The differences between the linear constraint values and
                their lower bounds.
            upper_diffs: The differences between the linear constraint values and
                their upper bounds.

        Returns:
            A tuple containing the transformed lower and upper differences.
        """
        if self._equation_scaling is not None:
            lower_diffs = lower_diffs.copy() * self._equation_scaling
            upper_diffs = upper_diffs.copy() * self._equation_scaling
        return lower_diffs, upper_diffs

    def init(self, mask: NDArray[np.bool_]) -> None:
        """Set the mask for the variable transform.

        This method allows setting a mask that indicates which variables are
        affected by this transform. The mask is a boolean array where `True`
        values indicate the variables that should be transformed, and `False`
        values indicate the variables that should remain unchanged.

        Args:
            mask: A boolean array indicating which variables are affected by this
                transform.
        """
        if self._scales is not None:
            self._scales = np.where(mask, self._scales, 1.0)
        if self._offsets is not None:
            self._offsets = np.where(mask, self._offsets, 0.0)


class DefaultObjectiveTransform(ObjectiveTransform):
    r"""Linearly scales objectives between domains.

    This class implements a linear transformation for objectives, allowing for
    scaling between the user-defined domain and the optimizer's internal domain.
    """

    def __init__(
        self,
        transform_config: ObjectiveTransformConfig,
    ) -> None:
        """Initialize the objective scaler.

        This scaler applies a linear scaling to objectives, defined in the
        transform configuration.

        Args:
            transform_config: The transform configuration.
        """
        self._scales: NDArray[np.float64] | None = transform_config.options.get(
            "scales", None
        )
        self._mask: NDArray[np.bool_] | None = None

    def to_optimizer(self, objectives: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform objective values to the optimizer domain.

        This method linearly scales objective values from the user-defined
        domain to the optimizer's internal domain.

        The input `objectives` may be a multi-dimensional array. It is assumed
        that the last axis of the array represents the objective values. If
        this is not the case, you must adjust the order of the axes before and
        after calling this method.

        Args:
            objectives: The objective values in the user domain to be transformed.

        Returns:
            The transformed objective values in the optimizer domain.
        """
        if self._scales is not None:
            return objectives / self._scales
        return objectives

    def from_optimizer(self, objectives: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform objective values to the user domain.

        This method scales objective values from the optimizer's internal domain
        back to the user-defined domain.

        The input `objectives` may be a multi-dimensional array. It is assumed
        that the last axis of the array represents the objective values. If
        this is not the case, you must adjust the order of the axes before and
        after calling this method.

        Args:
            objectives: The objective values in the optimizer domain to be transformed.

        Returns:
            The transformed objective values in the user domain.
        """
        if self._scales is not None:
            return objectives * self._scales
        return objectives

    def update(self, scales: ArrayLike) -> None:
        """Set the scaling factors for the objective transform.

        This method allows updating the scaling factors after the transform has
        been initialized. This can be useful in cases where the appropriate
        scaling factors are not known at initialization time and need to be
        determined based on information obtained during the optimization process.

        Args:
            scales: The new scaling factors to be applied to the objectives.
        """
        self._scales = np.asarray(scales, dtype=np.float64)
        if self._mask is not None:
            self._scales = np.where(self._mask, self._scales, 1.0)

    def init(self, mask: NDArray[np.bool_]) -> None:
        """Set the mask for the objective transform.

        This method allows setting a mask that indicates which objectives are
        affected by this transform. The mask is a boolean array where `True`
        values indicate the objectives that should be transformed, and `False`
        values indicate the objectives that should remain unchanged.

        Args:
            mask: A boolean array indicating which objectives are affected by this
                transform.
        """
        if self._scales is not None:
            self._scales = np.where(mask, self._scales, 1.0)
        self._mask = mask


class DefaultNonlinearConstraintTransform(NonLinearConstraintTransform):
    r"""Linearly scales constraints between domains.

    This class implements a linear transformation for constraints, allowing for
    scaling between the user-defined domain and the optimizer's internal domain.
    """

    def __init__(
        self,
        transform_config: NonlinearConstraintTransformConfig,
    ) -> None:
        """Initialize the constraint scaler.

        This scaler applies a linear scaling to objectives, defined in the
        transform configuration.

        Args:
            transform_config: The transform configuration.
        """
        self._scales: NDArray[np.float64] | None = transform_config.options.get(
            "scales", None
        )
        self._mask: NDArray[np.bool_] | None = None

    def to_optimizer(self, constraints: NDArray[np.float64]) -> NDArray[np.float64]:
        """Scales constraint values to the optimizer domain.

        This method maps nonlinear constraint values from the user-defined
        domain to the optimizer's internal domain.

        The input `constraints` may be a multi-dimensional array. It is assumed
        that the last axis of the array represents the constraint values. If
        this is not the case, you must adjust the order of the axes before and
        after calling this method.

        Args:
            constraints: The nonlinear constraint values in the user domain to
                be transformed.

        Returns:
            The transformed nonlinear constraint values in the optimizer domain.
        """
        if self._scales is not None:
            return constraints / self._scales
        return constraints

    def from_optimizer(self, constraints: NDArray[np.float64]) -> NDArray[np.float64]:
        """Scale constraint values to the user domain.

        This method scales nonlinear constraint values from the optimizer's
        internal domain back to the user-defined domain.

        The input `constraints` may be a multi-dimensional array. It is assumed
        that the last axis of the array represents the constraint values. If
        this is not the case, you must adjust the order of the axes before and
        after calling this method.

        Args:
            constraints: The nonlinear constraint values in the optimizer domain
                to be transformed.

        Returns:
            The transformed nonlinear constraint values in the user domain.
        """
        if self._scales is not None:
            return constraints * self._scales
        return constraints

    def bounds_to_optimizer(
        self, lower_bounds: NDArray[np.float64], upper_bounds: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Scale the right-hand-side bounds to the optimizer domain.

        This method scales the lower and upper bounds of the nonlinear
        constraints from the user-defined domain to the optimizer's internal
        domain. This scaling is necessary to ensure that the constraints remain
        valid after the variables have been transformed.

        Args:
            lower_bounds: The lower bounds on the right-hand-side values in the
                user domain.
            upper_bounds: The upper bounds on the right-hand-side values in the
                user domain.

        Returns:
            A tuple containing the transformed bounds.
        """
        if self._scales is not None:
            return lower_bounds / self._scales, upper_bounds / self._scales
        return lower_bounds, upper_bounds

    def nonlinear_constraint_diffs_from_optimizer(
        self, lower_diffs: NDArray[np.float64], upper_diffs: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Scale nonlinear constraint differences to the user domain.

        This method scales the differences between nonlinear constraint
        values and their lower/upper bounds from the optimizer's internal
        domain back to the user-defined domain. These differences are used to
        report constraint violations.

        Args:
            lower_diffs: The differences between the nonlinear constraint values
                and their lower bounds.
            upper_diffs: The differences between the nonlinear constraint values
                and their upper bounds.

        Returns:
            A tuple containing the transformed lower and upper differences.
        """
        if self._scales is not None:
            return lower_diffs * self._scales, upper_diffs * self._scales
        return lower_diffs, upper_diffs

    def update(self, scales: ArrayLike) -> None:
        """Set the scaling factors for the objective transform.

        This method allows updating the scaling factors after the transform has
        been initialized. This can be useful in cases where the appropriate
        scaling factors are not known at initialization time and need to be
        determined based on information obtained during the optimization process.

        Args:
            scales: The new scaling factors to be applied to the objectives.
        """
        self._scales = np.asarray(scales, dtype=np.float64)
        if self._mask is not None:
            self._scales = np.where(self._mask, self._scales, 1.0)

    def init(self, mask: NDArray[np.bool_]) -> None:
        """Set the mask for the constraint transform.

        This method allows setting a mask that indicates which constraints are
        affected by this transform. The mask is a boolean array where `True`
        values indicate the constraints that should be transformed, and `False`
        values indicate the constraints that should remain unchanged.

        Args:
            mask: A boolean array indicating which objectives are affected by this
                transform.
        """
        if self._scales is not None:
            self._scales = np.where(mask, self._scales, 1.0)
        self._mask = mask
