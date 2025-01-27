from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from ._results import Results

if TYPE_CHECKING:
    from ropt.transforms._transforms import Transforms

    from ._gradient_evaluations import GradientEvaluations
    from ._gradients import Gradients
    from ._realizations import Realizations

TypeResults = TypeVar("TypeResults", bound="Results")


@dataclass(slots=True)
class GradientResults(Results):
    """The `GradientResults` class stores gradient related results.

    This contains  the following additional information:

    1. The results of the function evaluations for perturbed variables.
    2. The parameters of the realizations, such as weights for objectives and
       constraints, and realization failures.
    3. The gradients of the calculated objectives and constraints.

    Attributes:
        evaluations:  Results of the function evaluations.
        realizations: The calculated parameters of the realizations.
        gradients:    The calculated gradients.
    """

    evaluations: GradientEvaluations
    realizations: Realizations
    gradients: Gradients | None

    def transform_back(self, transforms: Transforms) -> GradientResults:
        """Apply backward transforms to the results.

        Args:
            transforms: The transforms to apply.

        Returns:
            The transformed results.
        """
        return GradientResults(
            plan_id=self.plan_id,
            batch_id=self.batch_id,
            metadata=self.metadata,
            evaluations=self.evaluations.transform_back(transforms),
            realizations=self.realizations,
            gradients=(
                None
                if self.gradients is None
                else self.gradients.transform_back(transforms)
            ),
        )
