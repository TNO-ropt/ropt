from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from ropt.enums import ResultAxisName

from ._result_field import ResultField
from ._utils import _immutable_copy

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@dataclass
class Realizations(ResultField):
    """This class stores information on the realizations.

    The `failed_realizations` field is a boolean array that indicates for
    each realization whether the evaluation was successful or not.

    Depending on the type of objective or constraint calculation, the weights
    used for the realizations may change during optimization. This class stores
    for each objective and constraint a vector of weight values.

    All fields are two-dimensional matrices, where the first axis index denotes
    the function or constraint. The second axis index denotes the realization.

    Args:
        failed_realizations: Failed realizations
        objective_weights:   Realization weights for the objectives
        constraint_weights:  Realization weights for the constraints
    """

    failed_realizations: NDArray[np.bool_] = field(
        metadata={
            "__axes__": (ResultAxisName.REALIZATION,),
        },
    )
    objective_weights: Optional[NDArray[np.float64]] = field(
        default=None,
        metadata={
            "__axes__": (
                ResultAxisName.OBJECTIVE,
                ResultAxisName.REALIZATION,
            ),
        },
    )
    constraint_weights: Optional[NDArray[np.float64]] = field(
        default=None,
        metadata={
            "__axes__": (
                ResultAxisName.NONLINEAR_CONSTRAINT,
                ResultAxisName.REALIZATION,
            ),
        },
    )

    def __post_init__(self) -> None:
        """Make all array fields immutable copies.

        # noqa
        """
        self.failed_realizations = _immutable_copy(self.failed_realizations)
        self.objective_weights = _immutable_copy(self.objective_weights)
        self.constraint_weights = _immutable_copy(self.constraint_weights)
