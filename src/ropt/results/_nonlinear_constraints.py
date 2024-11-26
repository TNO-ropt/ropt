from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np  # noqa: TC002

from ropt import utils
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
    3. Scaled versions for these values.

    Attributes:
        values:            Non-linear constraint values.
        violations:        Violations of the nonlinear constraints.
        scaled_values:     Optional scaled non-linear constraint values.
        scaled_violations: Optional scaled violations of the nonlinear constraints.
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
    scaled_values: NDArray[np.float64] | None = field(
        default=None,
        metadata={
            "__axes__": (ResultAxis.NONLINEAR_CONSTRAINT,),
        },
    )
    scaled_violations: NDArray[np.float64] | None = field(
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
        self.scaled_values = _immutable_copy(self.scaled_values)
        self.scaled_violations = _immutable_copy(self.scaled_violations)

    @classmethod
    def create(
        cls,
        config: EnOptConfig,
        functions: Functions | None,
        constraint_auto_scales: NDArray[np.float64] | None,
    ) -> NonlinearConstraints | None:
        """Add constraint information.

        This factory function creates a `NonlinearConstraints` object with the
        correct information on nonlinear constraints, based on the given
        evaluations and functions.

        Args:
            config:                 The ensemble optimizer configuration object.
            functions:              An instance of the Function class.
            constraint_auto_scales: Optional constraint scales.

        Returns:
            A newly created NonlinearConstraints object.
        """
        if functions is None:
            return None

        def _get_scaled(
            unscaled: NDArray[np.float64] | None,
        ) -> NDArray[np.float64] | None:
            if unscaled is not None:
                assert config is not None
                return utils.scaling.scale_constraints(
                    config,
                    unscaled,
                    None if constraint_auto_scales is None else constraint_auto_scales,
                    axis=-1,
                )
            return None

        if config.nonlinear_constraints is not None:
            assert functions.constraints is not None
            values = _get_nonlinear_constraint_values(
                config, functions.constraints, axis=-1
            )
            violations = _get_nonlinear_constraint_violations(config, values)
            scaled_values = _get_scaled(values)
            scaled_violations = _get_scaled(violations)
            return NonlinearConstraints(
                values=values,
                violations=violations,
                scaled_values=scaled_values,
                scaled_violations=scaled_violations,
            )

        return None
