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
    """Results of a gradient evaluation batch.

    See [Working with Results](../optimizer_setup/results.md) for usage details.

    Attributes:
        evaluations:  Perturbed-variable evaluation data.
        realizations: Realization activity and weights.
        gradients:    Aggregated gradient values, or `None` if estimation failed.
    """

    evaluations: GradientEvaluations
    realizations: Realizations
    gradients: Gradients | None

    def unscale(self, context: EnOptContext) -> GradientResults:
        """Unscale the results.

        Unscales the sub-fields that carry values (`evaluations` and
        `gradients` when present). Realization metadata is passed through
        unchanged.

        Args:
            context: The context used by the source of the results.

        Returns:
            The unscaled results.
        """
        evaluations = self.evaluations._unscale(context)  # ruff: ignore[private-member-access]
        gradients: Gradients | None = self.gradients
        if self.gradients is not None:
            gradients = self.gradients._unscale(context)  # ruff: ignore[private-member-access]

        if evaluations is None and gradients is None:
            return self

        return GradientResults(
            batch_id=self.batch_id,
            metadata=self.metadata,
            names=self.names,
            evaluations=self.evaluations if evaluations is None else evaluations,
            realizations=self.realizations,
            gradients=self.gradients if gradients is None else gradients,
        )
