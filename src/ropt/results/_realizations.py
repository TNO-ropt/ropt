from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ropt.enums import ResultAxis

from ._result_field import ResultField
from ._utils import _immutable_copy

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@dataclass(slots=True)
class Realizations(ResultField):
    """Stores information about the realizations.

    The `Realizations` class stores data related to the individual realizations
    used in the optimization process. This includes:

    * **Failed Realizations:** A boolean array indicating whether each
      realization's evaluation was successful. `True` indicates a failed
      realization, while `False` indicates a successful one.
    * **Objective Weights:** A two-dimensional array of weights used for each
      objective in each realization. The first axis corresponds to the
      objectives, and the second axis corresponds to the realizations. These
      weights may change during optimization, depending on the type of objective
      calculation.
    * **Constraint Weights:** A two-dimensional array of weights used for each
      constraint in each realization. The first axis corresponds to the
      constraints, and the second axis corresponds to the realizations. These
      weights may change during optimization, depending on the type of
      constraint calculation.

    Attributes:
        failed_realizations: Boolean array indicating failed realizations.
        objective_weights:   Weights for each objective in each realization.
        constraint_weights:  Weights for each constraint in each realization.
    """

    failed_realizations: NDArray[np.bool_] = field(
        metadata={
            "__axes__": (ResultAxis.REALIZATION,),
        },
    )
    objective_weights: NDArray[np.float64] | None = field(
        default=None,
        metadata={
            "__axes__": (
                ResultAxis.OBJECTIVE,
                ResultAxis.REALIZATION,
            ),
        },
    )
    constraint_weights: NDArray[np.float64] | None = field(
        default=None,
        metadata={
            "__axes__": (
                ResultAxis.NONLINEAR_CONSTRAINT,
                ResultAxis.REALIZATION,
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
