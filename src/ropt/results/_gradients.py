from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ropt.enums import AxisName

from ._result_field import ResultField
from ._utils import _immutable_copy

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ropt.config import EnOptConfig


@dataclass(slots=True)
class Gradients(ResultField):
    """Stores the calculated objective and constraint gradients.

    The `Gradients` class stores the calculated gradients of the objective and
    constraint functions. These gradients are typically derived from function
    evaluations across all realizations, often through a process like averaging.
    The optimizer may handle multiple objectives and constraints. Multiple
    objective gradients are combined into a single vector, which is stored in
    the `target_objective` field. This is the gradient used by the optimizer.
    Multiple constraint gradients are handled individually by the optimizer.

    **Result descriptions**

    === "Weighted Objective Gradient"

        `target_objective`: The gradient of the target objective with
        respect to each variable:

        - Shape: $(n_v,)$, where:
            - $n_v$ is the number of variables.
        - Axis type:
            - [`AxisName.VARIABLE`][ropt.enums.AxisName.VARIABLE]

    === "Objective  Gradients"

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
        constraints:      The gradient of each individual constraint.
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
        """Make all array fields immutable copies.

        # noqa
        """
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
        """Create a Gradients object with the given information.

        Args:
            target_objective: The gradient of the target objective.
            objectives:       The objective gradients for each realization.
            constraints:      The constraint gradients for each realization.

        Returns:
            A new Functions object.
        """
        return Gradients(
            target_objective=target_objective,
            objectives=objectives,
            constraints=constraints,
        )

    def transform_from_optimizer(self, config: EnOptConfig) -> Gradients:
        """Apply transformations from optimizer space.

        Args:
            config:     The configuration used by the source of the results.

        Returns:
            The transformed results.
        """
        if (
            not config.objective_transforms
            and not config.nonlinear_constraint_transforms
        ):
            return self

        objectives = self.objectives
        constraints = self.constraints
        for objective_transform in config.objective_transforms:
            objectives = np.moveaxis(objectives, 0, -1)
            objectives = objective_transform.from_optimizer(objectives)
            objectives = np.moveaxis(objectives, 0, -1)
        if constraints is not None:
            for constraint_transform in config.nonlinear_constraint_transforms:
                constraints = np.moveaxis(constraints, 0, -1)
                constraints = constraint_transform.from_optimizer(constraints)
                constraints = np.moveaxis(constraints, 0, -1)

        return Gradients(
            target_objective=self.target_objective,
            objectives=objectives,
            constraints=constraints,
        )
