from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ropt.enums import AxisName

from ._result_field import ResultField
from ._utils import _immutable_copy

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@dataclass(slots=True)
class Realizations(ResultField):
    """Stores information about the realizations.

    The `Realizations` class stores data related to the individual realizations
    used in the optimization process.

    !!! info "Fields"

        === "Active Realizations"

            `active_realizations`: A boolean array indicating whether each
            realization's evaluation was evaluated. `True` indicates that a
            realization was evaluated:

            - Shape $(n_r,)$, where:
                - $n_r$ is the number of realizations.
            - Axis type:
                - [`AxisName.REALIZATION`][ropt.enums.AxisName.REALIZATION]

        === "Failed Realizations"

            `failed_realizations`: A boolean array indicating whether each
            realization's evaluation was successful. `True` indicates a failed
            realization, while `False` indicates a successful one:

            - Shape $(n_r,)$, where:
                - $n_r$ is the number of realizations.
            - Axis type:
                - [`AxisName.REALIZATION`][ropt.enums.AxisName.REALIZATION]

        === "Objective Weights"

            `objective_weights`: A two-dimensional array of weights used for
            each objective in each realization:

            - Shape $(n_o, n_r)$, where:
                - $n_o$ is the number of objectives.
                - $n_r$ is the number of realizations.
            - Axis types:
                - [`AxisName.OBJECTIVE`][ropt.enums.AxisName.OBJECTIVE]
                - [`AxisName.REALIZATION`][ropt.enums.AxisName.REALIZATION]

            These weights may change during optimization, depending on the type of
            objective calculation

        === "Constraint Weights"

            `constraint_weights`: A two-dimensional array of weights used for
            each constraint in each realization:

            - Shape $(n_c, n_r)$, where:
                - $n_c$ is the number of constraints.
                - $n_r$ is the number of realizations.
            - Axis types:
                - [`AxisName.NONLINEAR_CONSTRAINT`][ropt.enums.AxisName.NONLINEAR_CONSTRAINT]
                - [`AxisName.REALIZATION`][ropt.enums.AxisName.REALIZATION]

            These weights may change during optimization, depending on the type of
            constraint calculation

    Attributes:
        active_realizations: Boolean array indicating active realizations.
        failed_realizations: Boolean array indicating failed realizations.
        objective_weights:   Weights for each objective in each realization.
        constraint_weights:  Weights for each constraint in each realization.
    """

    active_realizations: NDArray[np.bool_] = field(
        metadata={
            "__axes__": (AxisName.REALIZATION,),
        },
    )
    failed_realizations: NDArray[np.bool_] = field(
        metadata={
            "__axes__": (AxisName.REALIZATION,),
        },
    )
    objective_weights: NDArray[np.float64] | None = field(
        default=None,
        metadata={
            "__axes__": (
                AxisName.OBJECTIVE,
                AxisName.REALIZATION,
            ),
        },
    )
    constraint_weights: NDArray[np.float64] | None = field(
        default=None,
        metadata={
            "__axes__": (
                AxisName.NONLINEAR_CONSTRAINT,
                AxisName.REALIZATION,
            ),
        },
    )

    def __post_init__(self) -> None:
        """Make all array fields immutable copies.

        # noqa
        """
        self.active_realizations = _immutable_copy(self.active_realizations)
        self.failed_realizations = _immutable_copy(self.failed_realizations)
        self.objective_weights = _immutable_copy(self.objective_weights)
        self.constraint_weights = _immutable_copy(self.constraint_weights)
