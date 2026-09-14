from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ropt._scaling import unscale_value
from ropt._utils import apply_direction
from ropt.enums import AxisName

from ._result_field import ResultField
from ._utils import _immutable_copy

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ropt.context import EnOptContext


@dataclass(slots=True)
class Functions(ResultField):
    """Aggregated objective and constraint function values.

    The same class carries both domains: the values the optimizer works with
    are found under `scaled`, the values as configured directly on the result.

    See [Working with Results](../optimizer_setup/results.md) for usage details.

    There is no target objective here: the quantity the optimizer minimizes is
    a weighted total over objectives that may differ in both scale and
    direction, so it has no counterpart in the configured domain. It is
    reported as `target_objective` on
    [`FunctionResults`][ropt.results.FunctionResults].

    **Result descriptions**

    === "Objectives"

        `objectives`: The calculated objective function values. This is a
        one-dimensional array of floating point values:

        - Shape $(n_o,)$, where:
            - $n_o$ is the number of objectives.
        - Axis type:
            - [`AxisName.OBJECTIVE`][ropt.enums.AxisName.OBJECTIVE]

    === "Constraints"

        `constraints`: The calculated constraint function values. This is a
        one-dimensional array of floating point values:

        - Shape $(n_c,)$, where:
            - $n_c$ is the number of constraints.
        - Axis type:
            - [`AxisName.NONLINEAR_CONSTRAINT`][ropt.enums.AxisName.NONLINEAR_CONSTRAINT]

    Attributes:
        objectives:  The value of each individual objective.
        constraints: The value of each individual constraint, if present.
    """

    objectives: NDArray[np.float64] = field(
        metadata={
            "__axes__": (AxisName.OBJECTIVE,),
        },
    )
    constraints: NDArray[np.float64] | None = field(
        default=None,
        metadata={
            "__axes__": (AxisName.NONLINEAR_CONSTRAINT,),
        },
    )

    def __post_init__(self) -> None:
        self.objectives = _immutable_copy(self.objectives)
        self.constraints = _immutable_copy(self.constraints)

    @classmethod
    def from_scaled(cls, context: EnOptContext, scaled: Functions) -> Functions:
        """Derive the configured function values from the scaled ones.

        Args:
            context: The context of the run.
            scaled:  The values as the optimizer sees them.

        Returns:
            A new `Functions` object.
        """
        # Undo the flip that made a maximized objective something to minimize,
        # so that the reported aggregate agrees in sign with the values it
        # summarizes.
        objectives = unscale_value(
            apply_direction(scaled.objectives, context.objectives.maximize),
            context.get_objective_scales(),
            context.get_objective_offsets(),
        )
        constraints = scaled.constraints
        if constraints is not None:
            constraint_scales = context.get_constraint_scales()
            assert constraint_scales is not None
            constraints = unscale_value(constraints, constraint_scales)
        return Functions(objectives=objectives, constraints=constraints)
