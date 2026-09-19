"""Configuration class for gradients."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    PositiveInt,
    model_validator,
)

from .constants import DEFAULT_NUMBER_OF_PERTURBATIONS


class GradientConfig(BaseModel):
    """Configuration class for gradient calculations.

    `GradientConfig` specifies how gradients are estimated in gradient-based
    optimizers. It is used as the `gradient` field of
    [`EnOptContext`][ropt.context.EnOptContext].

    See [Configuration Sections](../optimizer_setup/configuration_sections.md#gradient) for
    detailed descriptions and usage examples.

    `number_of_perturbations` defaults to
    [`DEFAULT_NUMBER_OF_PERTURBATIONS`][ropt.config.constants.DEFAULT_NUMBER_OF_PERTURBATIONS],
    and `perturbation_min_success` to `number_of_perturbations`.

    Attributes:
        number_of_perturbations:  Number of perturbations.
        perturbation_min_success: Minimum number of successful perturbations.
        merge_realizations:       Merge realizations for the final gradient.
        evaluation_policy:        How to evaluate functions and gradients.
    """

    number_of_perturbations: PositiveInt = DEFAULT_NUMBER_OF_PERTURBATIONS
    perturbation_min_success: PositiveInt | None = None
    merge_realizations: bool = False
    evaluation_policy: Literal["speculative", "separate", "auto"] = "auto"

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_default=True,
        frozen=True,
    )

    @model_validator(mode="after")
    def _check_perturbation_min_success(self) -> Self:
        perturbation_min_success = self.perturbation_min_success
        if (
            perturbation_min_success is None
            or perturbation_min_success > self.number_of_perturbations
        ):
            perturbation_min_success = self.number_of_perturbations
        return self.model_copy(
            update={"perturbation_min_success": perturbation_min_success}
        )
