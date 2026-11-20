from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ropt.enums import AxisName

from ._gradient_aggregates import _unscale_gradients
from ._result_field import ResultField
from ._utils import _immutable_copy

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ropt.context import EnOptContext


@dataclass(slots=True)
class GradientEstimates(ResultField):
    """Objective and constraint gradients estimated for each realization.

    These are the gradients that are aggregated over the realizations into
    [`GradientAggregates`][ropt.results.GradientAggregates]. A realization whose
    gradient was not estimated, because it failed, was not evaluated, or has
    zero weight, holds `NaN`.

    See [Working with Results](../results/results.md) for usage details.

    **Result descriptions**

    === "Objective Gradients"

        `objectives`: The gradients of each objective with respect to each
        variable, for each realization. This is a three-dimensional array of
        floating point values:

        - Shape $(n_r, n_o, n_v)$, where:
            - $n_r$ is the number of realizations.
            - $n_o$ is the number of objectives.
            - $n_v$ is the number of variables.
        - Axis types:
            - [`AxisName.REALIZATION`][ropt.enums.AxisName.REALIZATION]
            - [`AxisName.OBJECTIVE`][ropt.enums.AxisName.OBJECTIVE]
            - [`AxisName.VARIABLE`][ropt.enums.AxisName.VARIABLE]

    === "Constraint Gradients"

        `constraints`: The gradients of each nonlinear constraint with respect
        to each variable, for each realization. This is a three-dimensional
        array of floating point values:

        - Shape $(n_r, n_c, n_v)$, where:
            - $n_r$ is the number of realizations.
            - $n_c$ is the number of constraints.
            - $n_v$ is the number of variables.
        - Axis types:
            - [`AxisName.REALIZATION`][ropt.enums.AxisName.REALIZATION]
            - [`AxisName.NONLINEAR_CONSTRAINT`][ropt.enums.AxisName.NONLINEAR_CONSTRAINT]
            - [`AxisName.VARIABLE`][ropt.enums.AxisName.VARIABLE]

    Attributes:
        objectives:  The gradient of each objective, per realization.
        constraints: The gradient of each constraint per realization, if present.
    """

    objectives: NDArray[np.float64] = field(
        metadata={
            "__axes__": (
                AxisName.REALIZATION,
                AxisName.OBJECTIVE,
                AxisName.VARIABLE,
            ),
        },
    )
    constraints: NDArray[np.float64] | None = field(
        default=None,
        metadata={
            "__axes__": (
                AxisName.REALIZATION,
                AxisName.NONLINEAR_CONSTRAINT,
                AxisName.VARIABLE,
            ),
        },
    )

    def __post_init__(self) -> None:
        self.objectives = _immutable_copy(self.objectives)
        self.constraints = _immutable_copy(self.constraints)

    @classmethod
    def from_scaled(
        cls, context: EnOptContext, scaled: GradientEstimates
    ) -> GradientEstimates:
        """Derive the configured gradients from the scaled ones.

        Args:
            context: The context of the optimization or evaluation.
            scaled:  The gradients as the optimizer sees them.

        Returns:
            A new `GradientEstimates` object.
        """
        objectives, constraints = _unscale_gradients(
            context, scaled.objectives, scaled.constraints
        )
        return GradientEstimates(objectives=objectives, constraints=constraints)
