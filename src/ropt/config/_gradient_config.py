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

    This class, `GradientConfig`, defines the configuration of gradient
    calculations. It is used in an [`EnOptConfig`][ropt.config.EnOptConfig]
    object as the `gradient` field to specify how gradients are calculated in
    gradient-based optimizers.

    Gradients are estimated using function values calculated from perturbed
    variables and the unperturbed variables. The `number_of_perturbations` field
    determines the number of perturbed variables used, which must be at least
    one.

    If function evaluations for some perturbed variables fail, the gradient may
    still be estimated as long as a minimum number of evaluations succeed. The
    `perturbation_min_success` field specifies this minimum. By default, it
    equals `number_of_perturbations`.

    Gradients are calculated for each realization individually and then combined
    into a total gradient. If `number_of_perturbations` is low, or even just
    one, individual gradient calculations may be unreliable. In this case,
    setting `merge_realizations` to `True` directs the optimizer to combine the
    results of all realizations directly into a single gradient estimate.

    The `evaluation_policy` option controls how and when objective functions and
    gradients are calculated. It accepts one of three string values:

    - `"speculative"`: Evaluate the gradient whenever the objective function
        is requested, even if the optimizer hasn't explicitly asked for the
        gradient at that point. This approach can potentially improve load
        balancing on HPC clusters by initiating gradient work earlier, though
        its effectiveness depends on whether the optimizer can utilize these
        speculatively computed gradients.
    - `"separate"`: Always launch function and gradient evaluations as
        distinct operations, even if the optimizer requests both simultaneously.
        This is particularly useful when employing realization filters (see
        [`RealizationFilterConfig`][ropt.config.RealizationFilterConfig]) that
        might disable certain realizations, as it can potentially reduce the
        number of gradient evaluations needed.
    - `"auto"`: Evaluate functions and/or gradients strictly according to the
        optimizer's requests. Calculations are performed only when the
        optimization algorithm explicitly requires them.

    Attributes:
        number_of_perturbations:  Number of perturbations (default:
            [`DEFAULT_NUMBER_OF_PERTURBATIONS`][ropt.config.constants.DEFAULT_NUMBER_OF_PERTURBATIONS]).
        perturbation_min_success: Minimum number of successful function evaluations
            for perturbed variables (default: equal to `number_of_perturbations`).
        merge_realizations:       Merge all realizations for the final gradient
            calculation (default: `False`).
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
    )

    @model_validator(mode="after")
    def _check_perturbation_min_success(self) -> Self:
        if (
            self.perturbation_min_success is None
            or self.perturbation_min_success > self.number_of_perturbations
        ):
            self.perturbation_min_success = self.number_of_perturbations
        return self
