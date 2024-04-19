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
class Gradients(ResultField):
    """This class stores the calculated objective and constraint gradients.

    The objective and constraint gradients used by the optimizer are calculated
    from their values evaluated for all realizations, for instance by averaging.
    There may be multiple objectives and constraints. Multiple objectives are
    handled by the optimizer using a weighted sum of the individual gradients,
    stored in the `weighted_objective` field. Multiple constraints are directly
    handled by the optimizer.

    Args:
        weighted_objective: The weighted sum of the objective gradients
        objectives:         The value of each objective gradient
        constraints:        The value of each constraint gradient
    """

    weighted_objective: NDArray[np.float64] = field(
        metadata={"__axes__": (ResultAxisName.VARIABLE,)},
    )
    objectives: NDArray[np.float64] = field(
        metadata={
            "__axes__": (
                ResultAxisName.OBJECTIVE,
                ResultAxisName.VARIABLE,
            ),
        },
    )
    constraints: Optional[NDArray[np.float64]] = field(
        default=None,
        metadata={
            "__axes__": (
                ResultAxisName.NONLINEAR_CONSTRAINT,
                ResultAxisName.VARIABLE,
            ),
        },
    )

    def __post_init__(self) -> None:
        """Make all array fields immutable copies.

        # noqa
        """
        self.weighted_objective = _immutable_copy(self.weighted_objective)
        self.objectives = _immutable_copy(self.objectives)
        self.constraints = _immutable_copy(self.constraints)
