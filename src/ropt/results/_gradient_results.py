from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from ._results import Results

if TYPE_CHECKING:
    from ropt.transforms._transforms import OptModelTransforms

    from ._gradient_evaluations import GradientEvaluations
    from ._gradients import Gradients
    from ._realizations import Realizations

TypeResults = TypeVar("TypeResults", bound="Results")


@dataclass(slots=True)
class GradientResults(Results):
    """Stores results related to gradient evaluations.

    The `GradientResults` class extends the base
    [`Results`][ropt.results.Results] class to store data specific to gradient
    evaluations. This includes:

    * **Evaluations:** The results of the function evaluations for perturbed
      variables, including the perturbed variable values, objective values, and
      constraint values for each realization and perturbation. See
      [`GradientEvaluations`][ropt.results.GradientEvaluations].
    * **Realizations:** Information about the realizations, such as weights for
      objectives and constraints, and whether each realization was successful.
      See [`Realizations`][ropt.results.Realizations].
    * **Gradients:** The calculated gradients of the objectives and constraints.
      See [`Gradients`][ropt.results.Gradients].

    Attributes:
        evaluations:  Results of the function evaluations for perturbed
                      variables.
        realizations: The calculated parameters of the realizations.
        gradients:    The calculated gradients.
    """

    evaluations: GradientEvaluations
    realizations: Realizations
    gradients: Gradients | None

    def transform_from_optimizer(
        self, transforms: OptModelTransforms
    ) -> GradientResults:
        """Apply transformations from optimizer space.

        Args:
            transforms: The transforms to apply.

        Returns:
            The transformed results.
        """
        return GradientResults(
            batch_id=self.batch_id,
            metadata=self.metadata,
            evaluations=self.evaluations.transform_from_optimizer(transforms),
            realizations=self.realizations,
            gradients=(
                None
                if self.gradients is None
                else self.gradients.transform_from_optimizer(transforms)
            ),
        )
