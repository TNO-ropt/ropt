from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from ._results import Results

if TYPE_CHECKING:
    from ropt.context import EnOptContext

    from ._gradient_evaluations import GradientEvaluations
    from ._gradients import Gradients
    from ._realizations import Realizations

TypeResults = TypeVar("TypeResults", bound="Results")


@dataclass(slots=True)
class GradientResults(Results):
    """Store results related to gradient evaluations.

    The `GradientResults` class extends the base
    [`Results`][ropt.results.Results] class to store data specific to gradient
    evaluations. This includes:

    1. **Evaluations:** The results of the function evaluations for perturbed
       variables, including the perturbed variable values, objective values, and
       constraint values for each realization and perturbation. See
       [`GradientEvaluations`][ropt.results.GradientEvaluations].

     2. **Realizations:** Information about realizations, such as weights for
         objectives and constraints and whether each realization was successful.
       See [`Realizations`][ropt.results.Realizations].

    3. **Gradients:** Calculated gradients of objectives and constraints.
       See [`Gradients`][ropt.results.Gradients].

    Attributes:
        evaluations:  Function-evaluation results for perturbed variables.
        realizations: Realization-specific parameters.
        gradients:    Calculated gradients, if available.
    """

    evaluations: GradientEvaluations
    realizations: Realizations
    gradients: Gradients | None

    def transform_from_optimizer(self, context: EnOptContext) -> GradientResults:
        """Transform results from optimizer space to user space.

        This applies inverse transformations to transformable sub-fields
        (`evaluations` and `gradients` when present). Realization metadata is
        passed through unchanged.

        Args:
            context: The context used by the source of the results.

        Returns:
            The transformed results.
        """
        return GradientResults(
            batch_id=self.batch_id,
            metadata=self.metadata,
            names=self.names,
            evaluations=self.evaluations.transform_from_optimizer(context),
            realizations=self.realizations,
            gradients=(
                None
                if self.gradients is None
                else self.gradients.transform_from_optimizer(context)
            ),
        )
