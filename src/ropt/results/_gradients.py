from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ropt._scaling import unscale_diff
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

    The same class carries both domains: the gradients the optimizer works with
    are found under `scaled`, differentiated with respect to the scaled
    variables, the gradients as configured directly on the result.

    See [Working with Results](../optimizer_setup/results.md) for usage details.

    There is no target gradient here: the quantity the optimizer descends is a
    weighted total over objectives that may differ in both scale and direction,
    so it has no counterpart in the configured domain. It is reported as
    `target_gradient` on [`GradientResults`][ropt.results.GradientResults].

    **Result descriptions**

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
        objectives:  The gradient of each individual objective.
        constraints: The gradient of each individual constraint, if present.
    """

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
        self.objectives = _immutable_copy(self.objectives)
        self.constraints = _immutable_copy(self.constraints)

    @classmethod
    def from_scaled(cls, context: EnOptContext, scaled: Gradients) -> Gradients:
        """Derive the configured gradients from the scaled ones.

        Args:
            context: The context of the run.
            scaled:  The gradients as the optimizer sees them.

        Returns:
            A new `Gradients` object.
        """
        # A gradient carries a function in its numerator and a variable in its
        # denominator, so both axes must be undone: the function axis comes
        # first here and takes a trailing axis to broadcast against the
        # variables, while the variable scales divide along the last axis.
        variable_scales = context.variables.scales
        objectives = (
            unscale_diff(
                apply_direction(
                    scaled.objectives, context.objectives.maximize[:, np.newaxis]
                ),
                context.get_objective_scales()[:, np.newaxis],
            )
            / variable_scales
        )
        constraints = scaled.constraints
        if constraints is not None:
            constraint_scales = context.get_constraint_scales()
            assert constraint_scales is not None
            constraints = (
                unscale_diff(constraints, constraint_scales[:, np.newaxis])
                / variable_scales
            )
        return Gradients(objectives=objectives, constraints=constraints)
