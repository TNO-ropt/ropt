"""This module defines the base classes for transforms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ropt.config import VariableTransformConfig


class VariableTransform(ABC):
    """Abstract base class for variable transformations.

    Variable transforms are configured as an ordered chain. A transform defines
    a single step of that chain; `ropt` applies it by calling `to_optimizer` on
    each transform in turn, and `from_optimizer` on each in reverse order.

    Subclasses must implement methods to transform variables and related
    quantities between user and optimizer domains:

    - `to_optimizer` / `from_optimizer`: map variable values.
    - `magnitudes_to_optimizer`: map perturbation magnitudes.
    - `bound_constraint_diffs_from_optimizer`: map bound-violation differences.
    - `set_free_mask`: restrict the transform to the free variables.

    Override `linear_constraints_to_optimizer` and
    `linear_constraints_diffs_from_optimizer` if linear constraints are used.

    All arrays use the last axis for the variable dimension.

    See [Transforms](../optimizer_setup/variable_transforms.md) for lifecycle and guidance.
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
    def set_free_mask(self, mask: NDArray[np.bool_]) -> None:
        """Restrict the transform to the free variables.

        Called once when the context is built. Implementations must act as the
        identity where `mask` is `False`, so that fixed variables keep their
        user-domain values and bounds.

        Args:
            mask: Boolean array (`True` = the variable is free).
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

        Must be consistent with the variable transform (for example, if variables
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

        Overriding this method is optional: implement it only if the transform
        must support linear constraints. The base implementation raises
        `NotImplementedError`.

        Args:
            coefficients: Coefficient matrix `A`.
            lower_bounds: Lower RHS bounds.
            upper_bounds: Upper RHS bounds.

        Returns:
            Tuple of (coefficients, lower_bounds, upper_bounds) in optimizer domain.

        Raises:
            NotImplementedError: If the transform does not support linear
                constraints.
        """  # ruff: ignore[docstring-extraneous-returns]
        msg = "This transformer does not support linear constraints."
        raise NotImplementedError(msg)

    def linear_constraints_diffs_from_optimizer(
        self, lower_diffs: NDArray[np.float64], upper_diffs: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Transform linear-constraint-violation differences to user domain.

        Used for reporting constraint violations in user-domain units.

        Overriding this method is optional: implement it only if the transform
        must support linear constraints. The base implementation raises
        `NotImplementedError`.

        Args:
            lower_diffs: Constraint value minus lower bound.
            upper_diffs: Constraint value minus upper bound.

        Returns:
            Tuple of (lower_diffs, upper_diffs) in user domain.

        Raises:
            NotImplementedError: If the transform does not support linear
                constraints.
        """  # ruff: ignore[docstring-extraneous-returns]
        msg = "This transformer does not support linear constraints."
        raise NotImplementedError(msg)
