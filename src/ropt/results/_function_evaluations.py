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
    """Stores the results of function evaluations.

    The `FunctionEvaluations` class stores the results of evaluating the
    objective and constraint functions for a set of variables. It includes
    the following information:

    * **Variables:** The vector of variable values at which the functions were
      evaluated.
    * **Objectives:** The calculated objective function values for each
      realization. This is a two-dimensional array where each row corresponds to
      a realization and each column corresponds to an objective.
    * **Constraints:** The calculated constraint function values for each
      realization. This is a two-dimensional array where each row corresponds to
      a realization and each column corresponds to a constraint.
    * **Evaluation Info:** Optional metadata associated with each realization,
      potentially provided by the evaluator. If provided, each value in the info
      dictionary must be a one-dimensional array with a length equal to the
      number of realizations.

    Attributes:
        variables:       The variable vector.
        objectives:      The objective function values for each realization.
        constraints:     The constraint function values for each realization.
        evaluation_info: Optional metadata for each evaluated realization.
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
