"""This module defines the base classes for transforms.

Transformers can be used to transform controls and functions before
handing them over to the optimizer, or to transform them back when
produced by the optimizer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class VariableTransform(ABC):
    """Abstract base class for variable transformers."""

    @abstractmethod
    def to_optimizer(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Implement the transformation to optimizer space.

        The values may consist of an array with multiple dimensions. It is
        assumed that the last axis contains the variable values. Should this
        method be used in a context where that is not the case, adjust the
        order of the axis accordingly before and after calling this method.

        Args:
            values: The values to be transformed.

        Returns:
            The transformed values.
        """

    @abstractmethod
    def from_optimizer(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Implement the transformation from optimizer space.

        The values may consist of an array with multiple dimensions. It is
        assumed that the last axis contains the variable values. Should this
        method be used in a context where that is not the case, adjust the
        order of the axis accordingly before and after calling this method.

        Args:
            values: The values to be transformed.

        Returns:
            The transformed values.
        """

    @abstractmethod
    def magnitudes_to_optimizer(
        self, values: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Implement the transformation of perturbation magnitudes.

        Args:
            values: The values to be transformed.

        Returns:
            The transformed values.
        """

    @abstractmethod
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
        msg = "This transformer does not support linear constraints."
        raise NotImplementedError(msg)

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
        msg = "This transformer does not support linear constraints."
        raise NotImplementedError(msg)


class ObjectiveTransform(ABC):
    """Abstract base class for objective transformers."""

    @abstractmethod
    def to_optimizer(self, objectives: NDArray[np.float64]) -> NDArray[np.float64]:
        """Implement the transformation to optimizer space.

        The values may consist of an array with multiple dimensions. It is
        assumed that the last axis contains the objective values. Should this
        method be used in a context where that is not the case, adjust the
        order of the axis accordingly before and after calling this method.

        Args:
            objectives:     The values to be transformed.

        Returns:
            The transformed values.
        """

    @abstractmethod
    def from_optimizer(self, objectives: NDArray[np.float64]) -> NDArray[np.float64]:
        """Implement the transformation from optimizer space.

        The values may consist of an array with multiple dimensions. It is
        assumed that the last axis contains the objective values. Should this
        method be used in a context where that is not the case, adjust the
        order of the axis accordingly before and after calling this method.

        Args:
            objectives: The values to be transformed.

        Returns:
            The transformed values.
        """

    def weighted_objective_from_optimizer(
        self, weighted_objective: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Transform the weighted objective value.

        The optimizer generates weighted objective values using transformed
        values. This method is called to apply a transformation when
        transforming from optimizer space. For example when the transformation
        of the objectives involves a sign change to implement maximization, this
        can be used to change the sign of the weighted objective values.

        Note:
            This function may be applied to the weighted objective itself,
            or to its gradient, hence the input may be a vector of values.

        Args:
            weighted_objective: The weighted objective to transform.

        Returns:
            The transformed weighted objective.
        """
        return weighted_objective


class NonLinearConstraintTransform(ABC):
    """Abstract base class for non-linear constraint transformers."""

    @abstractmethod
    def bounds_to_optimizer(
        self, lower_bounds: NDArray[np.float64], upper_bounds: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Implement the of the right-hand side bounds to optimizer space.

        Args:
            lower_bounds: The lower bounds on the right-hand side values.
            upper_bounds: The upper bounds on the right-hand side values.

        Returns:
            The transformed right-hand side values.
        """

    @abstractmethod
    def to_optimizer(self, constraints: NDArray[np.float64]) -> NDArray[np.float64]:
        """Implement the transformation to optimizer space.

        The values may consist of an array with multiple dimensions. It is
        assumed that the last axis contains the objective values. Should this
        method be used in a context where that is not the case, adjust the
        order of the axis accordingly before and after calling this method.

        Args:
            constraints:     The values to be transformed.

        Returns:
            The transformed values.
        """

    @abstractmethod
    def from_optimizer(self, constraints: NDArray[np.float64]) -> NDArray[np.float64]:
        """Implement the transformation from optimizer space.

        The values may consist of an array with multiple dimensions. It is
        assumed that the last axis contains the objective values. Should this
        method be used in a context where that is not the case, adjust the
        order of the axis accordingly before and after calling this method.

        Args:
            constraints: The values to be transformed.

        Returns:
            The transformed values.
        """

    @abstractmethod
    def nonlinear_constraint_diffs_from_optimizer(
        self, lower_diffs: NDArray[np.float64], upper_diffs: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Implement the transformation of bound constraint diffs from optimizer space.

        Args:
            lower_diffs: The lower differences to be transformed.
            upper_diffs: The upper differences to be transformed.

        Returns:
            The transformed values.
        """
