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
class GradientEvaluations(ResultField):
    """Stores the results of evaluations for gradient calculations.

    The `GradientEvaluations` class stores the results of evaluating the
    objective and constraint functions for perturbed variables, which is
    necessary for gradient calculations. It contains the following information:

    * **Variables:** The vector of unperturbed variable values.
    * **Perturbed Variables:** A three-dimensional array of perturbed variable
      values. The axes represent (in order): realization, perturbation, and
      variable.
    * **Perturbed Objectives:** The calculated objective function values for
      each realization and perturbation. This is a three-dimensional array where
      the axes represent (in order): realization, perturbation, and objective.
    * **Perturbed Constraints:** The calculated constraint function values for
      each realization and perturbation. This is a three-dimensional array where
      the axes represent (in order): realization, perturbation, and constraint.
    * **Evaluation Info:** Optional metadata associated with each realization
      and perturbation, potentially provided by the evaluator. If provided, each
      value in the `evaluation_info` dictionary must be a two-dimensional array
      where the rows correspond to perturbations and the second columns
      correspond to realizations.


    Attributes:
        variables:             The unperturbed variable vector.
        perturbed_variables:   The perturbed variable values for each
                               realization and perturbation.
        perturbed_objectives:  The objective function values for each
                               realization and perturbation.
        perturbed_constraints: The constraint function values for each
                               realization and perturbation.
        evaluation_info:       Optional metadata for each evaluated
                               realization and perturbation.
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
    evaluation_info: dict[str, NDArray[Any]] = field(
        default_factory=dict,
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

    @classmethod
    def create(
        cls,
        variables: NDArray[np.float64],
        perturbed_variables: NDArray[np.float64],
        perturbed_objectives: NDArray[np.float64],
        perturbed_constraints: NDArray[np.float64] | None = None,
        evaluation_info: dict[str, NDArray[Any]] | None = None,
    ) -> GradientEvaluations:
        """Create a FunctionEvaluations object with the given information.

        Args:
            variables:             The unperturbed variable vector.
            perturbed_variables:   The unperturbed variable vector.
            perturbed_objectives:  The objective functions for each realization.
            perturbed_constraints: The constraint functions for each realization.
            evaluation_info:       Optional info for each evaluation.

        Returns:
            A new FunctionEvaluations object.
        """
        return GradientEvaluations(
            variables=variables,
            perturbed_variables=perturbed_variables,
            perturbed_objectives=perturbed_objectives,
            perturbed_constraints=perturbed_constraints,
            evaluation_info={} if evaluation_info is None else evaluation_info,
        )

    def transform_from_optimizer(
        self, transforms: OptModelTransforms
    ) -> GradientEvaluations:
        """Apply transformations from optimizer space.

        Args:
            transforms: The transforms to apply.

        Returns:
            The transformed results.
        """
        return GradientEvaluations(
            variables=(
                self.variables
                if transforms.variables is None
                else transforms.variables.from_optimizer(self.variables)
            ),
            perturbed_variables=(
                self.perturbed_variables
                if transforms.variables is None
                else transforms.variables.from_optimizer(self.perturbed_variables)
            ),
            perturbed_objectives=(
                self.perturbed_objectives
                if transforms.objectives is None
                else transforms.objectives.from_optimizer(self.perturbed_objectives)
            ),
            perturbed_constraints=(
                self.perturbed_constraints
                if (
                    self.perturbed_constraints is None
                    or transforms.nonlinear_constraints is None
                )
                else transforms.nonlinear_constraints.from_optimizer(
                    self.perturbed_constraints
                )
            ),
            evaluation_info=self.evaluation_info,
        )
