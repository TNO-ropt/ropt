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
    """Per-realization weights used to aggregate the evaluations.

    See [Working with Results](../results/results.md) for usage details.

    **Result descriptions**

    === "Objective Weights"

        `objective_weights`: A two-dimensional array of weights used for each
        objective in each realization:

        - Shape $(n_r, n_o)$, where:
            - $n_r$ is the number of realizations.
            - $n_o$ is the number of objectives.
        - Axis types:
            - [`AxisName.REALIZATION`][ropt.enums.AxisName.REALIZATION]
            - [`AxisName.OBJECTIVE`][ropt.enums.AxisName.OBJECTIVE]

        These weights may change during optimization, depending on the type of
        objective calculation.

    === "Constraint Weights"

        `constraint_weights`: A two-dimensional array of weights used for each
        constraint in each realization:

        - Shape $(n_r, n_c)$, where:
            - $n_r$ is the number of realizations.
            - $n_c$ is the number of constraints.
        - Axis types:
            - [`AxisName.REALIZATION`][ropt.enums.AxisName.REALIZATION]
            - [`AxisName.NONLINEAR_CONSTRAINT`][ropt.enums.AxisName.NONLINEAR_CONSTRAINT]

        These weights may change during optimization, depending on the type of
        constraint calculation.

    Attributes:
        objective_weights:  Objective weights per realization, if available.
        constraint_weights: Constraint weights per realization, if available.
    """

    objective_weights: NDArray[np.float64] | None = field(
        default=None,
        metadata={
            "__axes__": (
                AxisName.REALIZATION,
                AxisName.OBJECTIVE,
            ),
        },
    )
    constraint_weights: NDArray[np.float64] | None = field(
        default=None,
        metadata={
            "__axes__": (
                AxisName.REALIZATION,
                AxisName.NONLINEAR_CONSTRAINT,
            ),
        },
    )

    def __post_init__(self) -> None:
        self.objective_weights = _immutable_copy(self.objective_weights)
        self.constraint_weights = _immutable_copy(self.constraint_weights)
