from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ropt.enums import ResultAxis

from ._result_field import ResultField
from ._utils import _immutable_copy

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ropt.transforms import OptModelTransforms


@dataclass(slots=True)
class Gradients(ResultField):
    """This class stores the calculated objective and constraint gradients.

    The objective and constraint gradients used by the optimizer are calculated
    from their values evaluated for all realizations, for instance by averaging.
    There may be multiple objectives and constraints. Multiple objectives are
    handled by the optimizer using a weighted sum of the individual gradients,
    stored in the `weighted_objective` field. Multiple constraints are directly
    handled by the optimizer.

    Attributes:
        weighted_objective: The weighted sum of the objective gradients.
        objectives:         The value of each objective gradient.
        constraints:        The value of each constraint gradient.
    """

    weighted_objective: NDArray[np.float64] = field(
        metadata={"__axes__": (ResultAxis.VARIABLE,)},
    )
    objectives: NDArray[np.float64] = field(
        metadata={
            "__axes__": (
                ResultAxis.OBJECTIVE,
                ResultAxis.VARIABLE,
            ),
        },
    )
    constraints: NDArray[np.float64] | None = field(
        default=None,
        metadata={
            "__axes__": (
                ResultAxis.NONLINEAR_CONSTRAINT,
                ResultAxis.VARIABLE,
            ),
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
    ) -> Gradients:
        """Create a Gradients object with the given information.

        Args:
            config:             Configuration object.
            weighted_objective: The weighted objective.
            objectives:         The objective gradients for each realization.
            constraints:        The constraint gradients for each realization.

        Returns:
            A new Functions object.
        """
        return Gradients(
            weighted_objective=weighted_objective,
            objectives=objectives,
            constraints=constraints,
        )

    def transform_from_optimizer(self, transforms: OptModelTransforms) -> Gradients:
        """Apply transformations from optimizer space.

        Args:
            transforms: The transforms to apply.

        Returns:
            The transformed results.
        """
        objectives = self.objectives
        if transforms.objectives is not None:
            objectives = np.moveaxis(self.objectives, 0, -1)
            objectives = transforms.objectives.from_optimizer(objectives)
            objectives = np.moveaxis(objectives, 0, -1)

        constraints: NDArray[np.float64] | None = self.constraints
        if (
            self.constraints is not None
            and transforms.nonlinear_constraints is not None
        ):
            constraints = np.moveaxis(self.constraints, 0, -1)
            constraints = transforms.nonlinear_constraints.from_optimizer(constraints)
            constraints = np.moveaxis(constraints, 0, -1)

        return Gradients(
            weighted_objective=(
                self.weighted_objective
                if transforms.objectives is None
                else transforms.objectives.weighted_objective_from_optimizer(
                    self.weighted_objective
                )
            ),
            objectives=objectives,
            constraints=constraints,
        )
