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
class GradientEvaluations(ResultField):
    """This class contains the results of evaluations for gradient calculation.

    This class stores the variables with the calculated objective and
    constraints gradients. It contains the following information:

    1. The vector of variables at which the functions are evaluated.
    2. A three-dimensional array of perturbed variables, with variable values
       arranged along the third axis. The second axis index indicates the
       perturbation number, whereas the first axis index represents the
       realization number.
    3. The objectives and constraints for each realization and perturbed
       variable vector:  A three-dimensional array, with the objective or
       constraint values arranged along the third axis. The second axis index
       indicates the perturbation number, whereas the first axis index
       represents the realization number.
    4. Optional evaluation IDs that may have been passed from the evaluator,
       identifying each calculated realization and perturbation.

    Attributes:
        variables:                The unperturbed variable vector.
        perturbed_variables:      The variables for each realization and perturbation.
        perturbed_objectives:     The objective functions for each realization and
                                  perturbation.
        perturbed_constraints:    The constraint functions for each realization and
                                  perturbation.
        perturbed_evaluation_ids: Optional id of each evaluated realization.
    """

    variables: NDArray[np.float64] = field(
        metadata={
            "__axes__": (ResultAxis.VARIABLE,),
        },
    )
    perturbed_variables: NDArray[np.float64] = field(
        metadata={
            "__axes__": (
                ResultAxis.REALIZATION,
                ResultAxis.PERTURBATION,
                ResultAxis.VARIABLE,
            ),
        },
    )
    perturbed_objectives: NDArray[np.float64] = field(
        metadata={
            "__axes__": (
                ResultAxis.REALIZATION,
                ResultAxis.PERTURBATION,
                ResultAxis.OBJECTIVE,
            ),
        },
    )
    perturbed_constraints: NDArray[np.float64] | None = field(
        default=None,
        metadata={
            "__axes__": (
                ResultAxis.REALIZATION,
                ResultAxis.PERTURBATION,
                ResultAxis.NONLINEAR_CONSTRAINT,
            ),
        },
    )
    perturbed_evaluation_ids: NDArray[np.intc] | None = field(
        default=None,
        metadata={
            "__axes__": (
                ResultAxis.REALIZATION,
                ResultAxis.PERTURBATION,
            ),
        },
    )

    def __post_init__(self) -> None:
        """Make all array fields immutable copies.

        # noqa
        """
        self.variables = _immutable_copy(self.variables)
        self.perturbed_variables = _immutable_copy(self.perturbed_variables)
        self.perturbed_objectives = _immutable_copy(self.perturbed_objectives)
        self.perturbed_constraints = _immutable_copy(self.perturbed_constraints)
        self.perturbed_evaluation_ids = _immutable_copy(self.perturbed_evaluation_ids)

    @classmethod
    def create(
        cls,
        variables: NDArray[np.float64],
        perturbed_variables: NDArray[np.float64],
        perturbed_objectives: NDArray[np.float64],
        perturbed_constraints: NDArray[np.float64] | None = None,
        perturbed_evaluation_ids: NDArray[np.intc] | None = None,
    ) -> GradientEvaluations:
        """Create a FunctionEvaluations object with the given information.

        Args:
            config:                   Configuration object.
            variables:                The unperturbed variable vector.
            perturbed_variables:      The unperturbed variable vector.
            perturbed_objectives:     The objective functions for each realization.
            perturbed_constraints:    The constraint functions for each realization.
            perturbed_evaluation_ids: Optional IDs of the objective calculations.

        Returns:
            A new FunctionEvaluations object.
        """
        return GradientEvaluations(
            variables=variables,
            perturbed_variables=perturbed_variables,
            perturbed_objectives=perturbed_objectives,
            perturbed_constraints=perturbed_constraints,
            perturbed_evaluation_ids=perturbed_evaluation_ids,
        )

    def transform_back(self, transforms: OptModelTransforms) -> GradientEvaluations:
        """Apply backward transforms to the results.

        Args:
            transforms: The transforms to apply.

        Returns:
            The transformed results.
        """
        return GradientEvaluations(
            variables=(
                self.variables
                if transforms.variables is None
                else transforms.variables.backward(self.variables)
            ),
            perturbed_variables=(
                self.perturbed_variables
                if transforms.variables is None
                else transforms.variables.backward(self.perturbed_variables)
            ),
            perturbed_objectives=(
                self.perturbed_objectives
                if transforms.objectives is None
                else transforms.objectives.backward(self.perturbed_objectives)
            ),
            perturbed_constraints=(
                self.perturbed_constraints
                if (
                    self.perturbed_constraints is None
                    or transforms.nonlinear_constraints is None
                )
                else transforms.nonlinear_constraints.backward(
                    self.perturbed_constraints
                )
            ),
            perturbed_evaluation_ids=self.perturbed_evaluation_ids,
        )
