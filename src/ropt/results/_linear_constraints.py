from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np  # noqa: TCH002

from ropt.enums import ResultAxisName

from ._result_field import ResultField
from ._utils import (
    _get_linear_constraint_values,
    _get_linear_constraint_violations,
    _immutable_copy,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ropt.config.enopt import EnOptConfig

    from ._function_evaluations import FunctionEvaluations


@dataclass
class LinearConstraints(ResultField):
    """This class stores constraint values and violations.

    The following information is stored:

    1. Differences between the linear constraints and their right-hand-side
       values.
    2. Violations of constraints, defined as the absolute value of the
       difference if the constraint is violated, or else zero.

    Args:
        values:     Linear constraint values
        violations: Violations of the linear constraints
    """

    values: Optional[NDArray[np.float64]] = field(
        default=None,
        metadata={
            "__axes__": (ResultAxisName.LINEAR_CONSTRAINT,),
        },
    )
    violations: Optional[NDArray[np.float64]] = field(
        default=None,
        metadata={
            "__axes__": (ResultAxisName.LINEAR_CONSTRAINT,),
        },
    )

    def __post_init__(self) -> None:
        """Make all array fields immutable copies.

        # noqa
        """
        self.values = _immutable_copy(self.values)
        self.violations = _immutable_copy(self.violations)

    @classmethod
    def create(
        cls, config: EnOptConfig, evaluations: FunctionEvaluations
    ) -> Optional[LinearConstraints]:
        """Add constraint information.

        This factory function creates a `LinearConstraints` object with the
        correct information on the linear constraints, based on the given
        evaluations and functions.

        Args:
            config:      The ensemble optimizer configuration object
            evaluations: An instance of the Evaluations class

        Returns:
            A newly created LinearConstraints object.
        """
        kwargs = {}
        if config.linear_constraints is not None:
            kwargs["values"] = _get_linear_constraint_values(
                config,
                (
                    evaluations.variables
                    if evaluations.scaled_variables is None
                    else evaluations.scaled_variables
                ),
                axis=-1,
            )
            kwargs["violations"] = _get_linear_constraint_violations(
                config, kwargs["values"]
            )
        return LinearConstraints(**kwargs) if kwargs else None
