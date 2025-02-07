from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ropt.enums import ResultAxis

from ._result_field import ResultField
from ._utils import (
    _get_lower_bound_constraint_values,
    _get_lower_bound_violations,
    _get_upper_bound_constraint_values,
    _get_upper_bound_violations,
    _immutable_copy,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ropt.config.enopt import EnOptConfig

    from ._function_evaluations import FunctionEvaluations


@dataclass(slots=True)
class BoundConstraints(ResultField):
    """This class stores bound constraint values and violations.

    The following information is stored:

    1. Differences between the current variable and the lower or upper bounds.
    2. Violations of constraints, defined as the absolute value of the
       difference if the constraint is violated, or else zero.

    Attributes:
        lower_values:            Lower bound differences.
        lower_violations:        Lower bound violations.
        upper_values:            Upper bound differences.
        upper_violations:        Upper bound violations.
    """

    lower_values: NDArray[np.float64] | None = field(
        default=None,
        metadata={
            "__axes__": (ResultAxis.VARIABLE,),
        },
    )
    lower_violations: NDArray[np.float64] | None = field(
        default=None,
        metadata={
            "__axes__": (ResultAxis.VARIABLE,),
        },
    )
    upper_values: NDArray[np.float64] | None = field(
        default=None,
        metadata={
            "__axes__": (ResultAxis.VARIABLE,),
        },
    )
    upper_violations: NDArray[np.float64] | None = field(
        default=None,
        metadata={
            "__axes__": (ResultAxis.VARIABLE,),
        },
    )

    def __post_init__(self) -> None:
        """Make all array fields immutable copies.

        # noqa
        """
        self.lower_values = _immutable_copy(self.lower_values)
        self.lower_violations = _immutable_copy(self.lower_violations)
        self.upper_values = _immutable_copy(self.upper_values)
        self.upper_violations = _immutable_copy(self.upper_violations)

    @classmethod
    def create(
        cls, config: EnOptConfig, evaluations: FunctionEvaluations
    ) -> BoundConstraints | None:
        """Add constraint information.

        This factory function creates a `BoundConstraints` object with the
        correct information on bound constraints, based on the given
        evaluations.

        Args:
            config:      The ensemble optimizer configuration object.
            evaluations: An instance of the Evaluations class.

        Returns:
            A newly created BoundConstraints object.
        """
        kwargs = {}
        if np.any(np.isfinite(config.variables.lower_bounds)):
            lower_values = _get_lower_bound_constraint_values(
                config, evaluations.variables, axis=-1
            )
            lower_violations = _get_lower_bound_violations(lower_values)
            kwargs.update(
                {
                    "lower_values": lower_values,
                    "lower_violations": lower_violations,
                },
            )
        if np.any(np.isfinite(config.variables.upper_bounds)):
            upper_values = _get_upper_bound_constraint_values(
                config, evaluations.variables, axis=-1
            )
            upper_violations = _get_upper_bound_violations(upper_values)
            kwargs.update(
                {
                    "upper_values": upper_values,
                    "upper_violations": upper_violations,
                },
            )
        return BoundConstraints(**kwargs) if kwargs else None
