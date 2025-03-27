"""This module defines the base classes for transforms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class VariableTransform(ABC):
    """Abstract base class for variable transformations.

    This class defines the interface for transforming variables between the
    user-defined domain and the optimizer's internal domain. Concrete
    implementations of this class handle the specific logic for each type of
    transformation.

    When implementing a variable transformation, the following aspects must be
    considered:

    - **Variable Value Transformation:** Mapping variable values between the
      user and optimizer domains. This is achieved by overriding the
      [`to_optimizer`][ropt.transforms.base.VariableTransform.to_optimizer]
      and
      [`from_optimizer`][ropt.transforms.base.VariableTransform.from_optimizer]
      methods.
    - **Perturbation Magnitude Transformation:** Stochastic gradient-based
      algorithms use perturbations with specified magnitudes (see
      [`perturbation_magnitudes`][ropt.config.enopt.GradientConfig]). These
      magnitudes are typically defined in the user domain and must be
      transformed to the optimizer domain using the
      [`magnitudes_to_optimizer`][ropt.transforms.base.VariableTransform.magnitudes_to_optimizer]
      method.
    - **Bound Constraint Difference Transformation:** To report violations of
      variable bounds, the differences between variable values and their
      lower/upper bounds must be transformed from the optimizer domain back
      to the user domain. This is done using the
      [`bound_constraint_diffs_from_optimizer`][ropt.transforms.base.VariableTransform.bound_constraint_diffs_from_optimizer]
      method.
    - **Linear Constraint Transformation:** Linear constraints are generally
      defined by coefficients and right-hand-side values in the user domain.
      These must be transformed to the optimizer domain using the
      [`linear_constraints_to_optimizer`][ropt.transforms.base.VariableTransform.linear_constraints_to_optimizer]
      method.
    - **Linear Constraint Difference Transformation:** To report violations of
      linear constraints, the differences between the linear constraint
      values and their right-hand-side values must be transformed back to the
      user domain. This is done using the
      [`linear_constraints_diffs_from_optimizer`][ropt.transforms.base.VariableTransform.linear_constraints_diffs_from_optimizer]
      method.
    """

    @abstractmethod
    def to_optimizer(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform values from the user domain to the optimizer domain.

        This method maps variable values from the user-defined domain to the
        optimizer's internal domain. This transformation might involve scaling,
        shifting, or other operations to improve the optimizer's performance.

        The input `values` may be a multi-dimensional array. It is assumed that
        the last axis of the array represents the variable values. If this is
        not the case, you must adjust the order of the axes before and after
        calling this method.

        Args:
            values: The variable values in the user domain to be transformed.

        Returns:
            The transformed variable values in the optimizer domain.
        """

    @abstractmethod
    def from_optimizer(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform values from the optimizer domain to the user domain.

        This method maps variable values from the optimizer's internal domain
        back to the user-defined domain. This transformation reverses any
        scaling, shifting, or other operations that were applied to improve
        the optimizer's performance.

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

    @abstractmethod
    def magnitudes_to_optimizer(
        self, values: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Transform perturbation magnitudes to the optimizer domain.

        This method transforms perturbation magnitudes, typically used in
        stochastic gradient-based algorithms, from the user-defined domain to
        the optimizer's internal domain. The transformation ensures that the
        perturbations are applied correctly in the optimizer's space, which may
        have different scaling or units than the user domain.

        For example, if variables are scaled down in the optimizer domain, the
        perturbation magnitudes should also be scaled down proportionally.

        Args:
            values: The perturbation magnitudes in the user domain to be transformed.

        Returns:
            The transformed perturbation magnitudes in the optimizer domain.
        """

    @abstractmethod
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

        Args:
            lower_diffs: The differences between the variable values and their
                lower bounds in the optimizer domain.
            upper_diffs: The differences between the variable values and their
                upper bounds in the optimizer domain.

        Returns:
            A tuple containing the transformed differences.
        """

    def linear_constraints_to_optimizer(
        self,
        coefficients: NDArray[np.float64],
        lower_bounds: NDArray[np.float64],
        upper_bounds: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Transform linear constraints from the user domain to the optimizer domain.

        This method transforms linear constraints, defined by their coefficients
        and right-hand-side bounds, from the user-defined domain to the
        optimizer's internal domain. This is essential to maintain the
        validity of the constraints after variable transformations.

        For instance, if variables are scaled or shifted in the optimizer
        domain, the coefficients and bounds of the linear constraints must be
        adjusted accordingly to ensure the constraints remain consistent.

        The linear constraints are defined by the equation `A * x = b`, where `A`
        is the coefficient matrix, `x` is the variable vector, and `b` represents
        the right-hand-side bounds.

        Args:
            coefficients: The coefficient matrix.
            lower_bounds: The lower bounds on the right-hand-side values.
            upper_bounds: The upper bounds on the right-hand-side values.

        Returns:
            A tuple containing the transformed coefficient matrix and bounds.
        """
        msg = "This transformer does not support linear constraints."
        raise NotImplementedError(msg)

    def linear_constraints_diffs_from_optimizer(
        self, lower_diffs: NDArray[np.float64], upper_diffs: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Transform linear constraint differences to the user domain.

        This method transforms the differences between linear constraint values
        and their lower/upper bounds from the optimizer's internal domain back
        to the user-defined domain. These differences are used to report
        constraint violations.

        For example, if linear constraints are scaled in the optimizer domain,
        the differences between the constraint values and their bounds must be
        scaled back to the user domain to accurately reflect the constraint
        violations in the user's original units.

        Args:
            lower_diffs: The differences between the linear constraint values and
                their lower bounds.
            upper_diffs: The differences between the linear constraint values and
                their upper bounds.

        Returns:
            A tuple containing the transformed lower and upper differences.
        """
        msg = "This transformer does not support linear constraints."
        raise NotImplementedError(msg)


class ObjectiveTransform(ABC):
    """Abstract base class for objective transformations.

    This class defines the interface for transforming objective values between
    the user-defined domain and the optimizer's internal domain. Concrete
    implementations of this class handle the specific logic for each type of
    objective transformation.

    When implementing an objective transformation, the following aspects must be
    considered:

    - **Objective Value Transformation:** Mapping objective values between the
      user and optimizer domains. This is achieved by overriding the
      [`to_optimizer`][ropt.transforms.base.ObjectiveTransform.to_optimizer]
      and
      [`from_optimizer`][ropt.transforms.base.ObjectiveTransform.from_optimizer]
      methods.
    - **Weighted Objective Transformation:** The optimizer works with a
      single, weighted objective value. If the transformation affects the
      weighted objective, the
      [`weighted_objective_from_optimizer`][ropt.transforms.base.ObjectiveTransform.weighted_objective_from_optimizer]
      method should be overridden to handle this.
    """

    @abstractmethod
    def to_optimizer(self, objectives: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform objective values to the optimizer domain.

        This method maps objective values from the user-defined domain to the
        optimizer's internal domain. This transformation might involve scaling,
        shifting, or other operations to improve the optimizer's performance.

        The input `objectives` may be a multi-dimensional array. It is assumed
        that the last axis of the array represents the objective values. If
        this is not the case, you must adjust the order of the axes before and
        after calling this method.

        Args:
            objectives: The objective values in the user domain to be transformed.

        Returns:
            The transformed objective values in the optimizer domain.
        """

    @abstractmethod
    def from_optimizer(self, objectives: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform objective values to the user domain.

        This method maps objective values from the optimizer's internal domain
        back to the user-defined domain. This transformation reverses any
        scaling, shifting, or other operations that were applied to improve
        the optimizer's performance.

        The input `objectives` may be a multi-dimensional array. It is assumed
        that the last axis of the array represents the objective values. If
        this is not the case, you must adjust the order of the axes before and
        after calling this method.

        Args:
            objectives: The objective values in the optimizer domain to be transformed.

        Returns:
            The transformed objective values in the user domain.
        """

    def weighted_objective_from_optimizer(
        self, weighted_objective: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Transform the weighted objective to the user domain.

        The optimizer uses a single, weighted objective value evaluated in the
        optimizer domain. This method reverses that transformation, mapping the
        weighted objective back to the user domain.

        For example, if the transformation to the optimizer domain involved a
        sign change to convert a maximization problem into a minimization
        problem, this method would change the sign back.

        Note:
            This method may be applied to the weighted objective itself or to
            its gradient. Therefore, the input may be a scalar or a vector of
            values.

        Args:
            weighted_objective: The weighted objective value(s) to transform.

        Returns:
            The transformed weighted objective value(s).
        """
        return weighted_objective


class NonLinearConstraintTransform(ABC):
    """Abstract base class for nonlinear constraint transformations.

    This class defines the interface for transforming nonlinear constraint
    values between the user-defined domain and the optimizer's internal
    domain. Concrete implementations of this class handle the specific logic
    for each type of nonlinear constraint transformation.

    When implementing a nonlinear constraint transformation, the following
    aspects must be considered:

    - **Constraint Value Transformation:** Mapping constraint values between the
      user and optimizer domains. This is achieved by overriding the
      [`to_optimizer`][ropt.transforms.base.NonLinearConstraintTransform.to_optimizer]
      and
      [`from_optimizer`][ropt.transforms.base.NonLinearConstraintTransform.from_optimizer]
      methods.
    - **Right-Hand-Side Bound Transformation:** Mapping the right-hand-side
      bounds of the constraints between the user and optimizer domains. This is
      achieved by overriding the
      [`bounds_to_optimizer`][ropt.transforms.base.NonLinearConstraintTransform.bounds_to_optimizer]
      method.
    - **Constraint Difference Transformation:** To report violations of
      nonlinear constraints, the differences between constraint values and their
      lower/upper bounds must be transformed from the optimizer domain back to
      the user domain. This is done using the
      [`nonlinear_constraint_diffs_from_optimizer`][ropt.transforms.base.NonLinearConstraintTransform.nonlinear_constraint_diffs_from_optimizer]
      method.
    """

    @abstractmethod
    def to_optimizer(self, constraints: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform constraint values to the optimizer domain.

        This method maps nonlinear constraint values from the user-defined
        domain to the optimizer's internal domain. This transformation might
        involve scaling, shifting, or other operations to improve the
        optimizer's performance.

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

    @abstractmethod
    def from_optimizer(self, constraints: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform constraint values to the user domain.

        This method maps nonlinear constraint values from the optimizer's
        internal domain back to the user-defined domain. This transformation
        reverses any scaling, shifting, or other operations that were applied
        to improve the optimizer's performance.

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

    @abstractmethod
    def bounds_to_optimizer(
        self, lower_bounds: NDArray[np.float64], upper_bounds: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Transform the right-hand-side bounds to the optimizer domain.

        This method transforms the lower and upper bounds of the nonlinear
        constraints from the user-defined domain to the optimizer's internal
        domain. This transformation is necessary to ensure that the
        constraints remain valid after the variables have been transformed.

        For example, if constraint values are scaled or shifted in the
        optimizer domain, the bounds must be adjusted accordingly.

        Args:
            lower_bounds: The lower bounds on the right-hand-side values in the
                user domain.
            upper_bounds: The upper bounds on the right-hand-side values in the
                user domain.

        Returns:
            A tuple containing the transformed bounds.
        """

    @abstractmethod
    def nonlinear_constraint_diffs_from_optimizer(
        self, lower_diffs: NDArray[np.float64], upper_diffs: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Transform nonlinear constraint differences to the user domain.

        This method transforms the differences between nonlinear constraint
        values and their lower/upper bounds from the optimizer's internal
        domain back to the user-defined domain. These differences are used to
        report constraint violations.

        For example, if constraint values are scaled in the optimizer domain,
        the differences between the constraint values and their bounds must be
        scaled back to the user domain to accurately reflect the constraint
        violations in the user's original units.

        Args:
            lower_diffs: The differences between the nonlinear constraint values
                and their lower bounds.
            upper_diffs: The differences between the nonlinear constraint values
                and their upper bounds.

        Returns:
            A tuple containing the transformed lower and upper differences.
        """
