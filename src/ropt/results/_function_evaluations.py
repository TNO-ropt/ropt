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
    3. Scaled versions of the variables if variable scaling was enabled.
    4. Scaled versions of the objective and constraint values if scaling was
       enabled.
    5. Optional evaluation IDs that may have been passed from the evaluator,
       identifying each calculated realization.

    Attributes:
        variables:          The unperturbed variable vector.
        objectives:         The objective functions for each realization.
        constraints:        The constraint functions for each realization.
        scaled_variables:   Optional variables after scaling.
        scaled_objectives:  Optional scaled objectives.
        scaled_constraints: Optional scaled constraints.
        evaluation_ids:     Optional id of each evaluated realization.
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
    scaled_variables: NDArray[np.float64] | None = field(
        default=None,
        metadata={
            "__axes__": (ResultAxis.VARIABLE,),
        },
    )
    scaled_objectives: NDArray[np.float64] | None = field(
        default=None,
        metadata={
            "__axes__": (
                ResultAxis.REALIZATION,
                ResultAxis.OBJECTIVE,
            ),
        },
    )
    scaled_constraints: NDArray[np.float64] | None = field(
        default=None,
        metadata={
            "__axes__": (
                ResultAxis.REALIZATION,
                ResultAxis.NONLINEAR_CONSTRAINT,
            ),
        },
    )
    evaluation_ids: NDArray[np.intc] | None = field(
        default=None,
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
        self.scaled_variables = _immutable_copy(self.scaled_variables)
        self.scaled_objectives = _immutable_copy(self.scaled_objectives)
        self.scaled_constraints = _immutable_copy(self.scaled_constraints)
        self.evaluation_ids = _immutable_copy(self.evaluation_ids)

    @classmethod
    def create(  # noqa: PLR0913
        cls,
        config: EnOptConfig,
        variables: NDArray[np.float64],
        objectives: NDArray[np.float64],
        constraints: NDArray[np.float64] | None = None,
        objective_auto_scales: NDArray[np.float64] | None = None,
        constraint_auto_scales: NDArray[np.float64] | None = None,
        evaluation_ids: NDArray[np.intc] | None = None,
    ) -> FunctionEvaluations:
        """Create a FunctionEvaluations object with the given information.

        Args:
            config:                 Configuration object.
            variables:              The unperturbed variable vector.
            objectives:             The objective functions for each realization.
            constraints:            The constraint functions for each realization.
            objective_auto_scales:  Objective auto-scaling information.
            constraint_auto_scales: Constraint auto-scaling information.
            evaluation_ids:         Optional IDs of the objective calculations.

        Returns:
            A new FunctionEvaluations object.
        """
        unscaled_variables = utils.scaling.scale_back_variables(
            config, variables, axis=-1
        )
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
        return FunctionEvaluations(
            variables=variables if unscaled_variables is None else unscaled_variables,
            objectives=objectives,
            constraints=constraints,
            scaled_variables=None if unscaled_variables is None else variables,
            scaled_objectives=scaled_objectives,
            scaled_constraints=scaled_constraints,
            evaluation_ids=evaluation_ids,
        )

    def transform_back(self, transforms: OptModelTransforms) -> FunctionEvaluations:
        """Apply backward transforms to the results.

        Args:
            transforms: The transforms to apply.

        Returns:
            The transformed results.
        """
        return FunctionEvaluations(
            variables=(
                self.variables
                if transforms.variables is None
                else transforms.variables.backward(self.variables)
            ),
            objectives=(
                self.objectives
                if transforms.objectives is None
                else transforms.objectives.backward(self.objectives)
            ),
            constraints=(
                self.constraints
                if self.constraints is None or transforms.nonlinear_constraints is None
                else transforms.nonlinear_constraints.backward(self.constraints)
            ),
            evaluation_ids=self.evaluation_ids,
        )
