from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np  # noqa: TC002

from ropt.enums import ResultAxis

from ._result_field import ResultField
from ._utils import (
    _get_nonlinear_constraint_values,
    _get_nonlinear_constraint_violations,
    _immutable_copy,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ropt.config.enopt import EnOptConfig

    from ._functions import Functions


@dataclass(slots=True)
class NonlinearConstraints(ResultField):
    """This class stores constraint values and violations.

    The following information is stored:

    1. Differences between the non-linear constraints and their right-hand-side
       values.
    2. Violations of constraints, defined as the absolute value of the
       difference if the constraint is violated, or else zero.

    Attributes:
        values:            Non-linear constraint values.
        violations:        Violations of the nonlinear constraints.
    """

    values: NDArray[np.float64] | None = field(
        default=None,
        metadata={"__axes__": (ResultAxis.NONLINEAR_CONSTRAINT,)},
    )
    violations: NDArray[np.float64] | None = field(
        default=None,
        metadata={
            "__axes__": (ResultAxis.NONLINEAR_CONSTRAINT,),
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
        cls, config: EnOptConfig, functions: Functions | None
    ) -> NonlinearConstraints | None:
        """Add constraint information.

        This factory function creates a `NonlinearConstraints` object with the
        correct information on nonlinear constraints, based on the given
        evaluations and functions.

        Args:
            config:    The ensemble optimizer configuration object.
            functions: An instance of the Function class.

        Returns:
            A newly created NonlinearConstraints object.
        """
        if functions is None:
            return None

        if config.nonlinear_constraints is not None:
            assert functions.constraints is not None
            values = _get_nonlinear_constraint_values(
                config, functions.constraints, axis=-1
            )
            violations = _get_nonlinear_constraint_violations(config, values)
            return NonlinearConstraints(
                values=values,
                violations=violations,
            )

        return None
