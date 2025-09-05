from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ropt.enums import AxisName

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
    necessary for gradient calculations.

    **Result descriptions**

    === "Variables"

        `variables`: The vector of unperturbed variable values:

        - Shape: $(n_v,)$, where:
            - $n_v$ is the number of variables.
        - Axis type:
            - [`AxisName.VARIABLE`][ropt.enums.AxisName.VARIABLE]

    === "Perturbed Variables"

        `perturbed_variables`: A three-dimensional array of perturbed variable
        values for each realization and perturbation:

        - Shape: $(n_r, n_p, n_v)$, where:
            - $n_r$ is the number of realizations.
            - $n_p$ is the number of perturbations.
            - $n_v$ is the number of variables.
        - Axis type:
            - [`AxisName.REALIZATION`][ropt.enums.AxisName.REALIZATION]
            - [`AxisName.PERTURBATION`][ropt.enums.AxisName.PERTURBATION]
            - [`AxisName.VARIABLE`][ropt.enums.AxisName.VARIABLE]

    === "Perturbed Objectives"

        `perturbed_objectives`: A three-dimensional array of perturbed
        calculated objective function values for each realization and
        perturbation:

        - Shape $(n_r, n_p, n_o)$, where:
            - $n_r$ is the number of realizations.
            - $n_p$ is the number of perturbations.
            - $n_o$ is the number of objectives.
        - Axis types:
            - [`AxisName.REALIZATION`][ropt.enums.AxisName.REALIZATION]
            - [`AxisName.PERTURBATION`][ropt.enums.AxisName.PERTURBATION]
            - [`AxisName.OBJECTIVE`][ropt.enums.AxisName.OBJECTIVE]

    === "Perturbed Constraints"

        `perturbed_constraints`: A three-dimensional array of perturbed
        calculated non-linear constraint values for each realization and
        perturbation:

        - Shape $(n_r, n_p, n_c)$, where:
            - $n_r$ is the number of realizations.
            - $n_p$ is the number of perturbations.
            - $n_c$ is the number of constraints.
        - Axis types:
            - [`AxisName.REALIZATION`][ropt.enums.AxisName.REALIZATION]
            - [`AxisName.PERTURBATION`][ropt.enums.AxisName.PERTURBATION]
            - [`AxisName.NONLINEAR_CONSTRAINT`][ropt.enums.AxisName.NONLINEAR_CONSTRAINT]

    === "Evaluation Info"

        `evaluation_info`: Optional metadata associated with each realization,
        potentially provided by the evaluator. If provided, each value in the
        info dictionary must be a two-dimensional array of arbitrary type
        supported by `numpy` (including objects):

        - Shape: $(n_r, n_p)$, where:
            - $n_r$ is the number of realizations.
            - $n_p$ is the number of perturbations.
        - Axis types:
            - [`AxisName.REALIZATION`][ropt.enums.AxisName.REALIZATION]
            - [`AxisName.PERTURBATION`][ropt.enums.AxisName.PERTURBATION]


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
            "__axes__": (AxisName.VARIABLE,),
        },
    )
    perturbed_variables: NDArray[np.float64] = field(
        metadata={
            "__axes__": (
                AxisName.REALIZATION,
                AxisName.PERTURBATION,
                AxisName.VARIABLE,
            ),
        },
    )
    perturbed_objectives: NDArray[np.float64] = field(
        metadata={
            "__axes__": (
                AxisName.REALIZATION,
                AxisName.PERTURBATION,
                AxisName.OBJECTIVE,
            ),
        },
    )
    perturbed_constraints: NDArray[np.float64] | None = field(
        default=None,
        metadata={
            "__axes__": (
                AxisName.REALIZATION,
                AxisName.PERTURBATION,
                AxisName.NONLINEAR_CONSTRAINT,
            ),
        },
    )
    evaluation_info: dict[str, NDArray[Any]] = field(
        default_factory=dict,
        metadata={
            "__axes__": (
                AxisName.REALIZATION,
                AxisName.PERTURBATION,
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
