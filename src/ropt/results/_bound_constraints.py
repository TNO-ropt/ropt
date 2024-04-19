from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

from ropt import utils
from ropt.enums import ResultAxisName

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


@dataclass
class BoundConstraints(ResultField):
    """This class stores bound constraint values and violations.

    The following information is stored:

    1. Differences between the current variable and the lower or upper bounds.
    2. Violations of constraints, defined as the absolute value of the
       difference if the constraint is violated, or else zero.
    3. If variable scaling is configured, unscaled versions of all values.

    Args:
        lower_values:              Lower bound differences
        lower_violations:          Lower bound violations
        upper_values:              Upper bound differences
        upper_violations:          Upper bound violations
        unscaled_lower_values:     Optional unscaled lower bound differences
        unscaled_lower_violations: Optional unscaled lower bound violations
        unscaled_upper_values:     Optional unscaled upper bound differences
        unscaled_upper_violations: Optional unscaled upper bound violations
    """

    lower_values: Optional[NDArray[np.float64]] = field(
        default=None,
        metadata={
            "__axes__": (ResultAxisName.VARIABLE,),
        },
    )
    lower_violations: Optional[NDArray[np.float64]] = field(
        default=None,
        metadata={
            "__axes__": (ResultAxisName.VARIABLE,),
        },
    )
    upper_values: Optional[NDArray[np.float64]] = field(
        default=None,
        metadata={
            "__axes__": (ResultAxisName.VARIABLE,),
        },
    )
    upper_violations: Optional[NDArray[np.float64]] = field(
        default=None,
        metadata={
            "__axes__": (ResultAxisName.VARIABLE,),
        },
    )
    unscaled_lower_values: Optional[NDArray[np.float64]] = field(
        default=None,
        metadata={
            "__axes__": (ResultAxisName.VARIABLE,),
        },
    )
    unscaled_lower_violations: Optional[NDArray[np.float64]] = field(
        default=None,
        metadata={
            "__axes__": (ResultAxisName.VARIABLE,),
        },
    )
    unscaled_upper_values: Optional[NDArray[np.float64]] = field(
        default=None,
        metadata={
            "__axes__": (ResultAxisName.VARIABLE,),
        },
    )
    unscaled_upper_violations: Optional[NDArray[np.float64]] = field(
        default=None,
        metadata={
            "__axes__": (ResultAxisName.VARIABLE,),
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
        self.unscaled_lower_values = _immutable_copy(self.unscaled_lower_values)
        self.unscaled_lower_violations = _immutable_copy(self.unscaled_lower_violations)
        self.unscaled_upper_values = _immutable_copy(self.unscaled_upper_values)
        self.unscaled_upper_violations = _immutable_copy(self.unscaled_upper_violations)

    @classmethod
    def create(
        cls, config: EnOptConfig, evaluations: FunctionEvaluations
    ) -> Optional[BoundConstraints]:
        """Add constraint information.

        This factory function creates a `BoundConstraints` object with the correct
        information on bound constraints, based on the
        given evaluations.

        Args:
            config:      The ensemble optimizer configuration object
            evaluations: An instance of the Evaluations class

        Returns:
            A newly created BoundConstraints object.
        """

        def _get_unscaled(
            scaled: Optional[NDArray[np.float64]],
        ) -> Optional[NDArray[np.float64]]:
            unscaled = None
            if scaled is not None:
                assert config is not None
                unscaled = utils.scaling.scale_back_variables(
                    config,
                    scaled,
                    correct_offsets=False,
                    axis=-1,
                )
                if unscaled is not None:
                    unscaled.setflags(write=False)
            return unscaled

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
                    "unscaled_lower_values": _get_unscaled(lower_values),
                    "unscaled_lower_violations": _get_unscaled(lower_violations),
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
                    "unscaled_upper_values": _get_unscaled(upper_values),
                    "unscaled_upper_violations": _get_unscaled(upper_violations),
                },
            )
        return BoundConstraints(**kwargs) if kwargs else None
