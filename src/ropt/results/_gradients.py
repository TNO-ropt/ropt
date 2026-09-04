from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ropt._scaling import diff_from_optimizer
from ropt._utils import apply_direction
from ropt.enums import AxisName

from ._result_field import ResultField
from ._utils import _immutable_copy

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ropt.context import EnOptContext


@dataclass(slots=True)
class Gradients(ResultField):
    """Aggregated objective and constraint gradients.

    See [Working with Results](../optimizer_setup/results.md) for usage details.

    **Result descriptions**

    === "Weighted Objective Gradient"

        `target_objective`: The gradient of the target objective with
        respect to each variable:

        - Shape: $(n_v,)$, where:
            - $n_v$ is the number of variables.
        - Axis type:
            - [`AxisName.VARIABLE`][ropt.enums.AxisName.VARIABLE]

    === "Objective Gradients"

        `objectives`: The calculated gradients of each objective with respect to
        each variable. This is a two-dimensional array of floating point values:

        - Shape $(n_o, n_v)$, where:
            - $n_o$ is the number of objectives.
            - $n_v$ is the number of variables.
        - Axis types:
            - [`AxisName.OBJECTIVE`][ropt.enums.AxisName.OBJECTIVE]
            - [`AxisName.VARIABLE`][ropt.enums.AxisName.VARIABLE]

    === "Constraint Gradients"

        `constraints`: The calculated gradients of each nonlinear constraint
        with respect to each variable. This is a two-dimensional array of
        floating point values:

        - Shape $(n_c, n_v)$, where:
            - $n_c$ is the number of constraints.
            - $n_v$ is the number of variables.
        - Axis types:
            - [`AxisName.NONLINEAR_CONSTRAINT`][ropt.enums.AxisName.NONLINEAR_CONSTRAINT]
            - [`AxisName.VARIABLE`][ropt.enums.AxisName.VARIABLE]

    Attributes:
        target_objective: The gradient of the target objective.
        objectives:       The gradient of each individual objective.
        constraints:      The gradient of each individual constraint, if present.
    """

    target_objective: NDArray[np.float64] = field(
        metadata={"__axes__": (AxisName.VARIABLE,)},
    )
    objectives: NDArray[np.float64] = field(
        metadata={
            "__axes__": (
                AxisName.OBJECTIVE,
                AxisName.VARIABLE,
            ),
        },
    )
    constraints: NDArray[np.float64] | None = field(
        default=None,
        metadata={
            "__axes__": (
                AxisName.NONLINEAR_CONSTRAINT,
                AxisName.VARIABLE,
            ),
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
    ) -> Gradients:
        """Create a `Gradients` object from pre-aggregated gradient values.

        Args:
            target_objective: The gradient of the target objective.
            objectives:       Objective gradients.
            constraints:      Constraint gradients.

        Returns:
            A new `Gradients` object.
        """
        return Gradients(
            target_objective=target_objective,
            objectives=objectives,
            constraints=constraints,
        )

    def _transform_from_optimizer(self, context: EnOptContext) -> Gradients | None:
        # The objective and constraint axis comes first here, so the scales are
        # given a trailing axis to broadcast against the variables.
        objectives = diff_from_optimizer(
            apply_direction(
                self.objectives, context.objectives.maximize[:, np.newaxis]
            ),
            context.get_objective_scales()[:, np.newaxis],
        )
        constraints = self.constraints
        if constraints is not None:
            constraint_scales = context.get_constraint_scales()
            assert constraint_scales is not None
            constraints = diff_from_optimizer(
                constraints, constraint_scales[:, np.newaxis]
            )

        return Gradients(
            target_objective=self.target_objective,
            objectives=objectives,
            constraints=constraints,
        )
