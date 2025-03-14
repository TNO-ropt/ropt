from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ropt.enums import ResultAxis

from ._result_field import ResultField
from ._utils import _immutable_copy

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ropt.transforms import OptModelTransforms


@dataclass(slots=True)
class FunctionEvaluations(ResultField):
    """This class contains the results of evaluations for function calculation.

    This class stores the variables with the calculated objective and constraint
    functions. It contains the following information:

    1. The vector of variables at which the functions are evaluated.
    2. The calculated objectives and constraints for each realization: A
       two-dimensional array, with the objective or constraint values arranged
       along the second axis. The first axis index indicates the realization
       number.
    3. Optional evaluation IDs that may have been passed from the evaluator,
       identifying each calculated realization.

    Attributes:
        variables:       The unperturbed variable vector.
        objectives:      The objective functions for each realization.
        constraints:     The constraint functions for each realization.
        evaluation_info: Optional info for each evaluated realization.
    """

    variables: NDArray[np.float64] = field(
        metadata={
            "__axes__": (ResultAxis.VARIABLE,),
        },
    )
    objectives: NDArray[np.float64] = field(
        metadata={
            "__axes__": (
                ResultAxis.REALIZATION,
                ResultAxis.OBJECTIVE,
            ),
        },
    )
    constraints: NDArray[np.float64] | None = field(
        default=None,
        metadata={
            "__axes__": (
                ResultAxis.REALIZATION,
                ResultAxis.NONLINEAR_CONSTRAINT,
            ),
        },
    )
    evaluation_info: dict[str, NDArray[Any]] = field(
        default_factory=dict,
        metadata={
            "__axes__": (ResultAxis.REALIZATION,),
        },
    )

    def __post_init__(self) -> None:
        """Make all array fields immutable copies.

        # noqa
        """
        self.variables = _immutable_copy(self.variables)
        self.objectives = _immutable_copy(self.objectives)
        self.constraints = _immutable_copy(self.constraints)

    @classmethod
    def create(
        cls,
        variables: NDArray[np.float64],
        objectives: NDArray[np.float64],
        constraints: NDArray[np.float64] | None = None,
        evaluation_info: dict[str, NDArray[Any]] | None = None,
    ) -> FunctionEvaluations:
        """Create a FunctionEvaluations object with the given information.

        Args:
            config:          Configuration object.
            variables:       The unperturbed variable vector.
            objectives:      The objective functions for each realization.
            constraints:     The constraint functions for each realization.
            evaluation_info: Optional info for each evaluation.

        Returns:
            A new FunctionEvaluations object.
        """
        return FunctionEvaluations(
            variables=variables,
            objectives=objectives,
            constraints=constraints,
            evaluation_info={} if evaluation_info is None else evaluation_info,
        )

    def transform_from_optimizer(
        self, transforms: OptModelTransforms
    ) -> FunctionEvaluations:
        """Apply transformations from optimizer space.

        Args:
            transforms: The transforms to apply.

        Returns:
            The transformed results.
        """
        return FunctionEvaluations(
            variables=(
                self.variables
                if transforms.variables is None
                else transforms.variables.from_optimizer(self.variables)
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
            evaluation_info=self.evaluation_info,
        )
