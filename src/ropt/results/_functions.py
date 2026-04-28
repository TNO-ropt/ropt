from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ropt.enums import AxisName

from ._result_field import ResultField
from ._utils import _immutable_copy

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ropt.context import EnOptContext


@dataclass(slots=True)
class Functions(ResultField):
    """Store calculated objective and constraint function values.

    The `Functions` class stores the calculated values of the objective and
    constraint functions. These values are typically derived from the
    evaluations performed across realizations, often through aggregation. The
    optimizer may handle multiple objectives and constraints. Multiple
    objectives are combined into a single target objective stored in
    `target_objective`. Multiple constraints are handled individually.


    **Result descriptions**

    === "Weighted Objective"

        `target_objective`: The overall objective calculated as a weighted sum
        over the, possibly transformed, objectives. This is a single floating
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

    def transform_from_optimizer(self, context: EnOptContext) -> Functions:
        """Transform values from optimizer space to user space.

        The target objective is left unchanged because it is the scalar value
        optimized by the optimizer. Objective and non-linear constraint arrays
        are mapped back to user space using the inverse transform chain.

        Args:
            context: The context used by the source of the results.

        Returns:
            The transformed results.
        """
        if (
            not context.objective_transforms
            and not context.nonlinear_constraint_transforms
        ):
            return self

        objectives = self.objectives
        constraints = self.constraints
        for objective_transform in context.objective_transforms:
            objectives = objective_transform.from_optimizer(objectives)
        if constraints is not None:
            for constraint_transform in context.nonlinear_constraint_transforms:
                constraints = constraint_transform.from_optimizer(constraints)

        return Functions(
            target_objective=self.target_objective,
            objectives=objectives,
            constraints=constraints,
        )
