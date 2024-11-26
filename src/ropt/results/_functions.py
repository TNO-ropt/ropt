from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ropt import utils
from ropt.enums import ResultAxis

from ._result_field import ResultField
from ._utils import _immutable_copy

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ropt.config.enopt import EnOptConfig


@dataclass(slots=True)
class Functions(ResultField):
    """Store the calculated objective and constraint functions.

    The objective and constraint functions used by the optimizer are calculated
    from their values evaluated for all realizations, for instance by averaging.
    There may be multiple objectives and constraints. Multiple objectives are
    handled by the optimizer using a weighted sum, stored in the
    `weighted_objective` field. Multiple constraints are directly handled by the
    optimizer.

    If scaling of objectives and/or constraint functions is enabled, also scaled
    versions of the objective or constraint values are stored.

    Attributes:
        weighted_objective: The weighted sum of the objectives.
        objectives:         The value of each objective.
        constraints:        The value of each constraint.
        scaled_objectives:  Optional scaled objectives.
        scaled_constraints: Optional scaled constraints.
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
    scaled_objectives: NDArray[np.float64] | None = field(
        default=None,
        metadata={
            "__axes__": (ResultAxis.OBJECTIVE,),
        },
    )
    scaled_constraints: NDArray[np.float64] | None = field(
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
        self.scaled_objectives = _immutable_copy(self.scaled_objectives)
        self.scaled_constraints = _immutable_copy(self.scaled_constraints)

    @classmethod
    def create(  # noqa: PLR0913
        cls,
        config: EnOptConfig,
        weighted_objective: NDArray[np.float64],
        objectives: NDArray[np.float64],
        constraints: NDArray[np.float64] | None = None,
        objective_auto_scales: NDArray[np.float64] | None = None,
        constraint_auto_scales: NDArray[np.float64] | None = None,
    ) -> Functions:
        """Create a Functions object with the given information.

        Args:
            config:                 Configuration object.
            weighted_objective:     The weighted objective.
            objectives:             The objective functions for each realization.
            constraints:            The constraint functions for each realization.
            objective_auto_scales:  Objective auto-scaling information.
            constraint_auto_scales: Constraint auto-scaling information.

        Returns:
            A new Functions object.
        """
        scaled_objectives = utils.scaling.scale_objectives(
            config,
            objectives,
            None if objective_auto_scales is None else objective_auto_scales,
            axis=-1,
        )
        scaled_constraints = utils.scaling.scale_constraints(
            config,
            constraints,
            None if constraint_auto_scales is None else constraint_auto_scales,
            axis=-1,
        )
        return Functions(
            weighted_objective=weighted_objective,
            objectives=objectives,
            constraints=constraints,
            scaled_objectives=scaled_objectives,
            scaled_constraints=scaled_constraints,
        )
