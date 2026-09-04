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

    See [Working with Results](../optimizer_setup/results.md) for usage details.

    **Result descriptions**

    === "Weighted Objective"

        `target_objective`: The overall objective calculated as a weighted sum
        over the, possibly scaled, objectives. This is a single floating
        point value. It is defined as a `numpy` array of dimensions 0, hence it
        has no axes:

        - Shape: $()$
        - Axis type: `None`

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
        target_objective: The target objective value used by the optimizer.
        objectives:       The value of each individual objective.
        constraints:      The value of each individual constraint, if present.
    """

    target_objective: NDArray[np.float64] = field(
        metadata={"__axes__": ()},
    )
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
        self.target_objective = _immutable_copy(self.target_objective)
        self.objectives = _immutable_copy(self.objectives)
        self.constraints = _immutable_copy(self.constraints)

    @classmethod
    def create(
        cls,
        target_objective: NDArray[np.float64],
        objectives: NDArray[np.float64],
        constraints: NDArray[np.float64] | None = None,
    ) -> Functions:
        """Create a `Functions` object from pre-aggregated function values.

        Args:
            target_objective: The target objective used by the optimizer.
            objectives:       Objective function values.
            constraints:      Constraint function values.

        Returns:
            A new Functions object.
        """
        return Functions(
            target_objective=target_objective,
            objectives=objectives,
            constraints=constraints,
        )

    def _unscale(self, context: EnOptContext) -> Functions | None:
        # Undo the flip that made a maximized objective something to minimize,
        # so that the reported aggregate agrees in sign with the values it
        # summarizes.
        objectives = unscale_value(
            apply_direction(self.objectives, context.objectives.maximize),
            context.get_objective_scales(),
        )
        constraints = self.constraints
        if constraints is not None:
            constraint_scales = context.get_constraint_scales()
            assert constraint_scales is not None
            constraints = unscale_value(constraints, constraint_scales)

        return Functions(
            target_objective=self.target_objective,
            objectives=objectives,
            constraints=constraints,
        )
