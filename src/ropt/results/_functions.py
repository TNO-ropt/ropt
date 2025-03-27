from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ropt.enums import ResultAxis

from ._result_field import ResultField
from ._utils import _immutable_copy

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ropt.transforms import OptModelTransforms


@dataclass(slots=True)
class Functions(ResultField):
    """Stores the calculated objective and constraint function values.

    The `Functions` class stores the calculated values of the objective and
    constraint functions. These values are typically derived from the
    evaluations performed across all realizations, often through a process like
    averaging. The optimizer may handle multiple objectives and constraints.
    Multiple objectives are combined into a single weighted sum, which is stored
    in the `weighted_objective` field. Multiple constraints are handled
    individually by the optimizer.

    Attributes:
        weighted_objective: The weighted sum of the objective values.
        objectives:         The value of each individual objective.
        constraints:        The value of each individual constraint.
    """

    weighted_objective: NDArray[np.float64] = field(
        metadata={"__axes__": ()},
    )
    objectives: NDArray[np.float64] = field(
        metadata={
            "__axes__": (ResultAxis.OBJECTIVE,),
        },
    )
    constraints: NDArray[np.float64] | None = field(
        default=None,
        metadata={
            "__axes__": (ResultAxis.NONLINEAR_CONSTRAINT,),
        },
    )

    def __post_init__(self) -> None:
        """Make all array fields immutable copies.

        # noqa
        """
        self.weighted_objective = _immutable_copy(self.weighted_objective)
        self.objectives = _immutable_copy(self.objectives)
        self.constraints = _immutable_copy(self.constraints)

    @classmethod
    def create(
        cls,
        weighted_objective: NDArray[np.float64],
        objectives: NDArray[np.float64],
        constraints: NDArray[np.float64] | None = None,
    ) -> Functions:
        """Create a Functions object with the given information.

        Args:
            weighted_objective:     The weighted objective.
            objectives:             The objective functions for each realization.
            constraints:            The constraint functions for each realization.

        Returns:
            A new Functions object.
        """
        return Functions(
            weighted_objective=weighted_objective,
            objectives=objectives,
            constraints=constraints,
        )

    def transform_from_optimizer(self, transforms: OptModelTransforms) -> Functions:
        """Apply transformations from optimizer space.

        Args:
            transforms: The transforms to apply.

        Returns:
            The transformed results.
        """
        if transforms.objectives is None and transforms.nonlinear_constraints is None:
            return self

        return Functions(
            weighted_objective=(
                self.weighted_objective
                if transforms.objectives is None
                else transforms.objectives.weighted_objective_from_optimizer(
                    self.weighted_objective
                )
            ),
            objectives=(
                self.objectives
                if transforms.objectives is None
                else transforms.objectives.from_optimizer(self.objectives)
            ),
            constraints=(
                self.constraints
                if self.constraints is None or transforms.nonlinear_constraints is None
                else transforms.nonlinear_constraints.from_optimizer(self.constraints)
            ),
        )
