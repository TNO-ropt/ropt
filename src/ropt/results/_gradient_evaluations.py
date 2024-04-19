from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from ropt.enums import ResultAxisName

from ._result_field import ResultField
from ._utils import _immutable_copy

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@dataclass
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

    Args:
        variables:             The unperturbed variable vector.
        perturbed_variables:   The variables for each realization and perturbation.
        perturbed_objectives:  The objective functions for each realization and
                               perturbation.
        perturbed_constraints: The constraint functions for each realization and
                               perturbation.
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
    perturbed_constraints: Optional[NDArray[np.float64]] = field(
        default=None,
        metadata={
            "__axes__": (
                ResultAxisName.REALIZATION,
                ResultAxisName.PERTURBATION,
                ResultAxisName.NONLINEAR_CONSTRAINT,
            ),
        },
    )
    perturbed_evaluation_ids: Optional[NDArray[np.intc]] = field(
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
        self.perturbed_evaluation_ids = _immutable_copy(self.perturbed_evaluation_ids)
