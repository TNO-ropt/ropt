"""This module defines the base classes for transforms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ropt.config import (
        NonlinearConstraintTransformConfig,
        ObjectiveTransformConfig,
        VariableTransformConfig,
    )


class VariableTransform(ABC):
    """Abstract base class for variable transformations.

    Subclasses must implement methods to transform variables and related
    quantities between user and optimizer domains:

    - `to_optimizer` / `from_optimizer`: map variable values.
    - `magnitudes_to_optimizer`: map perturbation magnitudes.
    - `bound_constraint_diffs_from_optimizer`: map bound-violation differences.
    - `init`: apply a mask selecting which variables this transform affects.

    Override `linear_constraints_to_optimizer` and
    `linear_constraints_diffs_from_optimizer` if linear constraints are used.

    All arrays use the last axis for the variable dimension.

    See [Transforms](../usage/transforms.md) for lifecycle and guidance.
    """

    @abstractmethod
    def __init__(
        self,
        transform_config: VariableTransformConfig,
    ) -> None:
        """Initialize the variable transform.

        Args:
            transform_config: The transform configuration.
        """

    @abstractmethod
    def init(self, mask: NDArray[np.bool_]) -> None:
        """Apply a mask selecting which variables this transform affects.

        The mask combines the free-variable mask with the per-transform
        assignment. Unmasked positions must pass through unchanged.

        Args:
            mask: Boolean array (`True` = this transform applies).
        """

    @abstractmethod
    def to_optimizer(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform variable values from user domain to optimizer domain.

        The last axis represents variables. Multi-dimensional arrays are
        supported.

        Args:
            values: Variable values in the user domain.

        Returns:
            Transformed values in the optimizer domain.
        """

    @abstractmethod
    def from_optimizer(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform variable values from optimizer domain to user domain.

        The last axis represents variables. Multi-dimensional arrays are
        supported.

        Args:
            values: Variable values in the optimizer domain.

        Returns:
            Transformed values in the user domain.
        """

    @abstractmethod
    def magnitudes_to_optimizer(
        self, values: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Transform perturbation magnitudes to the optimizer domain.

        Must be consistent with the variable transform (e.g., if variables
        are divided by scale, magnitudes should be too).

        Args:
            values: Perturbation magnitudes in the user domain.

        Returns:
            Magnitudes in the optimizer domain.
        """

    @abstractmethod
    def bound_constraint_diffs_from_optimizer(
        self, lower_diffs: NDArray[np.float64], upper_diffs: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Transform bound-violation differences to the user domain.

        Used for reporting constraint violations in user-domain units.

        Args:
            lower_diffs: Variable value minus lower bound (optimizer domain).
            upper_diffs: Variable value minus upper bound (optimizer domain).

        Returns:
            Tuple of (lower_diffs, upper_diffs) in user domain.
        """

    def linear_constraints_to_optimizer(
        self,
        coefficients: NDArray[np.float64],
        lower_bounds: NDArray[np.float64],
        upper_bounds: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Transform linear constraint coefficients and bounds to optimizer domain.

        Adjusts the coefficient matrix and RHS bounds so that linear constraints
        remain valid after the variable transformation.

        Args:
            coefficients: Coefficient matrix `A`.
            lower_bounds: Lower RHS bounds.
            upper_bounds: Upper RHS bounds.

        Returns:
            Tuple of (coefficients, lower_bounds, upper_bounds) in optimizer domain.
        """  # noqa: DOC202
        msg = "This transformer does not support linear constraints."
        raise NotImplementedError(msg)

    def linear_constraints_diffs_from_optimizer(
        self, lower_diffs: NDArray[np.float64], upper_diffs: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Transform linear-constraint-violation differences to user domain.

        Used for reporting constraint violations in user-domain units.

        Args:
            lower_diffs: Constraint value minus lower bound.
            upper_diffs: Constraint value minus upper bound.

        Returns:
            Tuple of (lower_diffs, upper_diffs) in user domain.
        """  # noqa: DOC202
        msg = "This transformer does not support linear constraints."
        raise NotImplementedError(msg)

    def update(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401, B027
        """Update internal state mid-run (optional).

        Override to support runtime parameter changes when values are not
        known at initialization.

        Args:
            args:   Positional arguments.
            kwargs: Keyword arguments.
        """


class ObjectiveTransform(ABC):
    """Abstract base class for objective transformations.

    Subclasses must implement `to_optimizer` and `from_optimizer` to map
    objective values between user and optimizer domains, plus `init` to
    apply a mask selecting which objectives this transform affects.

    All arrays use the last axis for the objective dimension.
    """

    @abstractmethod
    def __init__(
        self,
        transform_config: ObjectiveTransformConfig,
    ) -> None:
        """Initialize the objective transform.

        Args:
            transform_config: The transform configuration.
        """

    @abstractmethod
    def init(self, mask: NDArray[np.bool_]) -> None:
        """Apply a mask selecting which objectives this transform affects.

        Unmasked positions must pass through unchanged.

        Args:
            mask: Boolean array (`True` = this transform applies).
        """

    @abstractmethod
    def to_optimizer(self, objectives: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform objective values from user domain to optimizer domain.

        The last axis represents objectives. Multi-dimensional arrays are
        supported.

        Args:
            objectives: Objective values in the user domain.

        Returns:
            Transformed values in the optimizer domain.
        """

    @abstractmethod
    def from_optimizer(self, objectives: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform objective values from optimizer domain to user domain.

        The last axis represents objectives. Multi-dimensional arrays are
        supported.

        Args:
            objectives: Objective values in the optimizer domain.

        Returns:
            Transformed values in the user domain.
        """

    def update(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401, B027
        """Update internal state mid-run (optional).

        Args:
            args:   Positional arguments.
            kwargs: Keyword arguments.
        """


class NonlinearConstraintTransform(ABC):
    """Abstract base class for nonlinear constraint transformations.

    Subclasses must implement:

    - `to_optimizer` / `from_optimizer`: map constraint values.
    - `bounds_to_optimizer`: map constraint RHS bounds.
    - `nonlinear_constraint_diffs_from_optimizer`: map violation differences.
    - `init`: apply a mask selecting which constraints this transform affects.

    All arrays use the last axis for the constraint dimension.
    """

    @abstractmethod
    def __init__(
        self,
        transform_config: NonlinearConstraintTransformConfig,
    ) -> None:
        """Initialize the constraint transform.

        Args:
            transform_config: The transform configuration.
        """

    @abstractmethod
    def init(self, mask: NDArray[np.bool_]) -> None:
        """Apply a mask selecting which constraints this transform affects.

        Unmasked positions must pass through unchanged.

        Args:
            mask: Boolean array (`True` = this transform applies).
        """

    @abstractmethod
    def to_optimizer(self, constraints: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform constraint values from user domain to optimizer domain.

        The last axis represents constraints. Multi-dimensional arrays are
        supported.

        Args:
            constraints: Constraint values in the user domain.

        Returns:
            Transformed values in the optimizer domain.
        """

    @abstractmethod
    def from_optimizer(self, constraints: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform constraint values from optimizer domain to user domain.

        The last axis represents constraints. Multi-dimensional arrays are
        supported.

        Args:
            constraints: Constraint values in the optimizer domain.

        Returns:
            Transformed values in the user domain.
        """

    @abstractmethod
    def bounds_to_optimizer(
        self, lower_bounds: NDArray[np.float64], upper_bounds: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Transform constraint RHS bounds to the optimizer domain.

        Adjusts bounds to remain consistent with the constraint transform.

        Args:
            lower_bounds: Lower RHS bounds in user domain.
            upper_bounds: Upper RHS bounds in user domain.

        Returns:
            Tuple of (lower_bounds, upper_bounds) in optimizer domain.
        """

    @abstractmethod
    def nonlinear_constraint_diffs_from_optimizer(
        self, lower_diffs: NDArray[np.float64], upper_diffs: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Transform constraint-violation differences to user domain.

        Used for reporting constraint violations in user-domain units.

        Args:
            lower_diffs: Constraint value minus lower bound.
            upper_diffs: Constraint value minus upper bound.

        Returns:
            Tuple of (lower_diffs, upper_diffs) in user domain.
        """

    def update(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401, B027
        """Update internal state mid-run (optional).

        Args:
            args:   Positional arguments.
            kwargs: Keyword arguments.
        """
