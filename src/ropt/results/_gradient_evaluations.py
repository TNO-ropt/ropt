from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ropt import utils
from ropt.enums import ResultAxisName

from ._result_field import ResultField
from ._utils import _immutable_copy

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ropt.config.enopt import EnOptConfig


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
    4. Scaled versions of the variables and perturbed variables if scaling
       was enabled.
    5. Scaled versions of the objective and constraint values if scaling was
       enabled.
    6. Optional evaluation IDs that may have been passed from the evaluator,
       identifying each calculated realization and perturbation.

    Attributes:
        variables:                    The unperturbed variable vector.
        perturbed_variables:          The variables for each realization and perturbation.
        perturbed_objectives:         The objective functions for each realization and
                                      perturbation.
        perturbed_constraints:        The constraint functions for each realization and
                                      perturbation.
        scaled_variables:             Optional variables after scaling back.
        scaled_perturbed_variables:   Optional variables after scaling back.
        scaled_perturbed_objectives:  Optional scaled objectives.
        scaled_perturbed_constraints: Optional scaled constraints.
        perturbed_evaluation_ids:     Optional id of each evaluated realization.
    """

    variables: NDArray[np.float64] = field(
        metadata={
            "__axes__": (ResultAxisName.VARIABLE,),
        },
    )
    perturbed_variables: NDArray[np.float64] = field(
        metadata={
            "__axes__": (
                ResultAxisName.REALIZATION,
                ResultAxisName.PERTURBATION,
                ResultAxisName.VARIABLE,
            ),
        },
    )
    perturbed_objectives: NDArray[np.float64] = field(
        metadata={
            "__axes__": (
                ResultAxisName.REALIZATION,
                ResultAxisName.PERTURBATION,
                ResultAxisName.OBJECTIVE,
            ),
        },
    )
    perturbed_constraints: NDArray[np.float64] | None = field(
        default=None,
        metadata={
            "__axes__": (
                ResultAxisName.REALIZATION,
                ResultAxisName.PERTURBATION,
                ResultAxisName.NONLINEAR_CONSTRAINT,
            ),
        },
    )
    scaled_variables: NDArray[np.float64] | None = field(
        default=None,
        metadata={
            "__axes__": (ResultAxisName.VARIABLE,),
        },
    )
    scaled_perturbed_variables: NDArray[np.float64] | None = field(
        default=None,
        metadata={
            "__axes__": (
                ResultAxisName.REALIZATION,
                ResultAxisName.PERTURBATION,
                ResultAxisName.VARIABLE,
            ),
        },
    )
    scaled_perturbed_objectives: NDArray[np.float64] | None = field(
        default=None,
        metadata={
            "__axes__": (
                ResultAxisName.REALIZATION,
                ResultAxisName.PERTURBATION,
                ResultAxisName.OBJECTIVE,
            ),
        },
    )
    scaled_perturbed_constraints: NDArray[np.float64] | None = field(
        default=None,
        metadata={
            "__axes__": (
                ResultAxisName.REALIZATION,
                ResultAxisName.PERTURBATION,
                ResultAxisName.NONLINEAR_CONSTRAINT,
            ),
        },
    )
    perturbed_evaluation_ids: NDArray[np.intc] | None = field(
        default=None,
        metadata={
            "__axes__": (
                ResultAxisName.REALIZATION,
                ResultAxisName.PERTURBATION,
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
        self.scaled_variables = _immutable_copy(self.scaled_variables)
        self.scaled_perturbed_variables = _immutable_copy(
            self.scaled_perturbed_variables
        )
        self.scaled_perturbed_objectives = _immutable_copy(
            self.scaled_perturbed_objectives
        )
        self.scaled_perturbed_constraints = _immutable_copy(
            self.scaled_perturbed_constraints
        )
        self.perturbed_evaluation_ids = _immutable_copy(self.perturbed_evaluation_ids)

    @classmethod
    def create(  # noqa: PLR0913
        cls,
        config: EnOptConfig,
        variables: NDArray[np.float64],
        perturbed_variables: NDArray[np.float64],
        perturbed_objectives: NDArray[np.float64],
        perturbed_constraints: NDArray[np.float64] | None = None,
        objective_auto_scales: NDArray[np.float64] | None = None,
        constraint_auto_scales: NDArray[np.float64] | None = None,
        perturbed_evaluation_ids: NDArray[np.intc] | None = None,
    ) -> GradientEvaluations:
        """Create a FunctionEvaluations object with the given information.

        Args:
            config:                   Configuration object.
            variables:                The unperturbed variable vector.
            perturbed_variables:      The unperturbed variable vector.
            perturbed_objectives:     The objective functions for each realization.
            perturbed_constraints:    The constraint functions for each realization.
            objective_auto_scales:    Objective auto-scaling information.
            constraint_auto_scales:   Constraint auto-scaling information.
            perturbed_evaluation_ids: Optional IDs of the objective calculations.

        Returns:
            A new FunctionEvaluations object.
        """
        unscaled_variables = utils.scaling.scale_back_variables(
            config, variables, axis=-1
        )
        unscaled_perturbed_variables = utils.scaling.scale_back_variables(
            config, perturbed_variables, axis=-1
        )
        scaled_perturbed_objectives = utils.scaling.scale_objectives(
            config,
            perturbed_objectives,
            None if objective_auto_scales is None else objective_auto_scales,
            axis=-1,
        )
        scaled_perturbed_constraints = utils.scaling.scale_constraints(
            config,
            perturbed_constraints,
            None if constraint_auto_scales is None else constraint_auto_scales,
            axis=-1,
        )
        return GradientEvaluations(
            variables=variables if unscaled_variables is None else unscaled_variables,
            perturbed_variables=(
                perturbed_variables
                if unscaled_perturbed_variables is None
                else unscaled_perturbed_variables
            ),
            perturbed_objectives=perturbed_objectives,
            perturbed_constraints=perturbed_constraints,
            scaled_variables=None if unscaled_variables is None else variables,
            scaled_perturbed_variables=(
                None if unscaled_perturbed_variables is None else perturbed_variables
            ),
            scaled_perturbed_objectives=scaled_perturbed_objectives,
            scaled_perturbed_constraints=scaled_perturbed_constraints,
            perturbed_evaluation_ids=perturbed_evaluation_ids,
        )
