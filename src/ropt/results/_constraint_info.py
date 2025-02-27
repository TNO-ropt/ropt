from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ropt.enums import ResultAxis

from ._result_field import ResultField
from ._utils import _immutable_copy

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ropt.config.enopt import EnOptConfig
    from ropt.transforms import OptModelTransforms


@dataclass(slots=True)
class ConstraintInfo(ResultField):
    """This class stores constraint differences.

    Attributes:
        bound_lower:     Lower bound differences.
        bound_upper:     Upper bound differences.
        linear_lower:    Lower linear constraint differences.
        linear_upper:    Upper Linear constraint differences.
        nonlinear_lower: Lower non-linear constraint differences.
        nonlinear_upper: Upper non-Linear constraint differences.
    """

    bound_lower: NDArray[np.float64] | None = field(
        default=None, metadata={"__axes__": (ResultAxis.VARIABLE,)}
    )
    bound_upper: NDArray[np.float64] | None = field(
        default=None, metadata={"__axes__": (ResultAxis.VARIABLE,)}
    )
    linear_lower: NDArray[np.float64] | None = field(
        default=None, metadata={"__axes__": (ResultAxis.LINEAR_CONSTRAINT,)}
    )
    linear_upper: NDArray[np.float64] | None = field(
        default=None, metadata={"__axes__": (ResultAxis.LINEAR_CONSTRAINT,)}
    )
    nonlinear_lower: NDArray[np.float64] | None = field(
        default=None, metadata={"__axes__": (ResultAxis.NONLINEAR_CONSTRAINT,)}
    )
    nonlinear_upper: NDArray[np.float64] | None = field(
        default=None, metadata={"__axes__": (ResultAxis.NONLINEAR_CONSTRAINT,)}
    )
    bound_violation: NDArray[np.float64] | None = field(
        default=None, metadata={"__axes__": (ResultAxis.VARIABLE,)}
    )
    linear_violation: NDArray[np.float64] | None = field(
        default=None, metadata={"__axes__": (ResultAxis.LINEAR_CONSTRAINT,)}
    )
    nonlinear_violation: NDArray[np.float64] | None = field(
        default=None, metadata={"__axes__": (ResultAxis.NONLINEAR_CONSTRAINT,)}
    )

    def __post_init__(self) -> None:
        """Make all array fields immutable copies.

        # noqa
        """
        self.bound_lower = _immutable_copy(self.bound_lower)
        self.bound_upper = _immutable_copy(self.bound_upper)
        self.linear_lower = _immutable_copy(self.linear_lower)
        self.linear_upper = _immutable_copy(self.linear_upper)
        self.nonlinear_lower = _immutable_copy(self.nonlinear_lower)
        self.nonlinear_upper = _immutable_copy(self.nonlinear_upper)
        if self.bound_lower is not None and self.bound_upper is not None:
            self.bound_violation = _immutable_copy(
                np.maximum(
                    np.where(self.bound_lower < 0.0, -self.bound_lower, 0.0),
                    np.where(self.bound_upper > 0.0, self.bound_upper, 0.0),
                )
            )
        if self.linear_lower is not None and self.linear_upper is not None:
            self.linear_violation = _immutable_copy(
                np.maximum(
                    np.where(self.linear_lower < 0.0, -self.linear_lower, 0.0),
                    np.where(self.linear_upper > 0.0, self.linear_upper, 0.0),
                )
            )
        if self.nonlinear_lower is not None and self.nonlinear_upper is not None:
            self.nonlinear_violation = _immutable_copy(
                np.maximum(
                    np.where(self.nonlinear_lower < 0.0, -self.nonlinear_lower, 0.0),
                    np.where(self.nonlinear_upper > 0.0, self.nonlinear_upper, 0.0),
                )
            )

    @classmethod
    def create(
        cls,
        config: EnOptConfig,
        variables: NDArray[np.float64],
        constraints: NDArray[np.float64] | None,
    ) -> ConstraintInfo | None:
        """Add constraint difference information.

        This stores the following constraint differences:
        1. The difference between variables and their lower and upper bounds.
        2. The difference between the linear constraints with their
           right-hand-side upper and lower bounds, calculated for given variables.
        3. The differences between non-linear constraint values and their
           right-hand-side upper and lower bounds.

        Args:
            config:      The ensemble optimizer configuration object.
            variables:   The variables to check.
            constraints: The constraints to check (optional).

        Returns:
            A newly created ConstraintInfo object or None.
        """
        diffs: dict[str, NDArray[np.float64] | None] = {}

        if np.all(np.isfinite(config.variables.lower_bounds)) or np.all(
            np.isfinite(config.variables.upper_bounds)
        ):
            diffs["bound_lower"] = variables - config.variables.lower_bounds
            diffs["bound_upper"] = variables - config.variables.upper_bounds

        if config.linear_constraints is not None:
            values = np.matmul(config.linear_constraints.coefficients, variables)
            diffs["linear_lower"] = values - config.linear_constraints.lower_bounds
            diffs["linear_upper"] = values - config.linear_constraints.upper_bounds

        if constraints is not None:
            assert config.nonlinear_constraints is not None
            diffs["nonlinear_lower"] = (
                constraints - config.nonlinear_constraints.lower_bounds
            )
            diffs["nonlinear_upper"] = (
                constraints - config.nonlinear_constraints.upper_bounds
            )

        if diffs:
            return ConstraintInfo(**diffs)

        return None

    def transform_from_optimizer(
        self, transforms: OptModelTransforms
    ) -> ConstraintInfo:
        if transforms.variables is None and transforms.nonlinear_constraints is None:
            return self

        diffs: dict[str, NDArray[np.float64] | None] = asdict(self)

        if transforms.variables is not None and self.bound_lower is not None:
            assert self.bound_upper is not None
            diffs["bound_lower"], diffs["bound_upper"] = (
                transforms.variables.bound_constraint_diffs_from_optimizer(
                    self.bound_lower, self.bound_upper
                )
            )

        if transforms.variables is not None and self.linear_lower is not None:
            assert self.linear_upper is not None
            diffs["linear_lower"], diffs["linear_upper"] = (
                transforms.variables.linear_constraints_diffs_from_optimizer(
                    self.linear_lower, self.linear_upper
                )
            )

        if transforms.nonlinear_constraints is not None:
            assert self.nonlinear_lower is not None
            assert self.nonlinear_upper is not None
            diffs["nonlinear_lower"], diffs["nonlinear_upper"] = (
                transforms.nonlinear_constraints.nonlinear_constraint_diffs_from_optimizer(
                    self.nonlinear_lower, self.nonlinear_upper
                )
            )

        return ConstraintInfo(**diffs)
