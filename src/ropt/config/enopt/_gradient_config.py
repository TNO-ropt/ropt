"""Configuration class for gradients."""

from __future__ import annotations

import sys
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, PositiveInt, model_validator

from ropt.config.utils import check_enum_values
from ropt.config.validated_types import Array1D, Array1DInt, ArrayEnum  # noqa: TCH001
from ropt.enums import BoundaryType, PerturbationType

from .constants import (
    DEFAULT_NUMBER_OF_PERTURBATIONS,
    DEFAULT_PERTURBATION_BOUNDARY_TYPE,
    DEFAULT_PERTURBATION_MAGNITUDE,
    DEFAULT_PERTURBATION_TYPE,
)

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


class GradientConfig(BaseModel):
    """The configuration class for gradient calculations.

    This class defines the configuration for gradient calculations, configured
    by the `gradients` field in an
    [`EnOptConfig`][ropt.config.enopt.EnOptConfig] object.

    If the optimizer requires gradient information, it is estimated from a set
    of function values calculated from perturbed variables and from the
    unperturbed variables. The number of perturbations is determined by the
    `number_of_perturbations` field, which should be at least one and may be
    larger than the number of variables.

    Some function evaluations for the perturbed variables may fail, for instance
    due to an error in a long-running simulation. As long as not too many
    evaluations fail, the gradient may still be estimated. The
    `perturbation_min_success` field determines how many perturbed variables
    should be successfully evaluated. By default, this parameter is set equal to
    the number of perturbations.

    Perturbations are generally produced by sampler objects configured in the
    parent [`EnOptConfig`][ropt.config.enopt.EnOptConfig] object. The `samplers`
    field contains, for each variable, an index into the tuple of these
    configured samplers, indicating which sampler should be used to generate
    perturbation values for that variable.

    The generated perturbation values are added to the unperturbed variables
    after multiplication with the perturbation magnitudes given by the
    `perturbation_magnitudes` field. The perturbation values may be used
    directly or first modified in various ways. The `perturbation_types` field
    determines if and how this is done for each variable (see the
    [`PerturbationType`][ropt.enums.PerturbationType] enumeration for details).

    The perturbed variables can occasionally violate the bound constraints
    defined for the variables. This may be undesirable, for instance if the
    function evaluation may fail for variables that violate these constraints.
    The `boundary_types` array determines what action is taken to rectify such a
    situation (see the [`BoundaryType`][ropt.enums.BoundaryType] enumeration for
    more details).

    Both the `perturbation_types` and `boundary_types` fields are defined as
    integer arrays, but their values are limited to the values of the
    [`PerturbationType`][ropt.enums.PerturbationType] and
    [`BoundaryType`][ropt.enums.BoundaryType] enumerations, respectively.

    The gradient is calculated for each realization individually, and the
    resulting gradients are afterwards combined into a total gradient. If the
    number of perturbations is low, the calculation of the individual gradients
    may be unreliable. In particular, in the case of a single perturbation, the
    result is likely inaccurate. In such a case, the `merge_realizations` flag
    can be set to direct the optimizer to use a different calculation to combine
    the results of all realizations directly into an estimation of the total
    gradient.

    Attributes:
        number_of_perturbations:  The number of perturbations (default:
            [`DEFAULT_NUMBER_OF_PERTURBATIONS`][ropt.config.enopt.constants.DEFAULT_NUMBER_OF_PERTURBATIONS])
        perturbation_min_success: The minimum number of successful function
                                  evaluations for perturbed variables (default:
                                  equal to the number of perturbations)
        perturbation_magnitudes:  The magnitudes of the perturbations for each variable
                                  (default:
            [`DEFAULT_PERTURBATION_MAGNITUDE`][ropt.config.enopt.constants.DEFAULT_PERTURBATION_MAGNITUDE])
        perturbation_types:       The type of perturbation for each variable
                                  ([`PerturbationType`][ropt.enums.PerturbationType],
                                  default:
            [`DEFAULT_PERTURBATION_TYPE`][ropt.config.enopt.constants.DEFAULT_PERTURBATION_TYPE])
        boundary_types:           How perturbations that violate boundary conditions
                                  are treated (see [`BoundaryType`][ropt.enums.BoundaryType]),
                                  default:
            [`DEFAULT_PERTURBATION_BOUNDARY_TYPE`][ropt.config.enopt.constants.DEFAULT_PERTURBATION_BOUNDARY_TYPE]).
        samplers:                 The index of the sampler to use for each variable
        merge_realizations:       If all realizations should be merged for the final
                                  gradient calculation (default: `False`)
    """

    number_of_perturbations: PositiveInt = DEFAULT_NUMBER_OF_PERTURBATIONS
    perturbation_min_success: Optional[PositiveInt] = None
    perturbation_magnitudes: Array1D = np.array(DEFAULT_PERTURBATION_MAGNITUDE)
    perturbation_types: ArrayEnum = np.array(DEFAULT_PERTURBATION_TYPE)
    boundary_types: ArrayEnum = np.array(DEFAULT_PERTURBATION_BOUNDARY_TYPE)
    samplers: Optional[Array1DInt] = None
    merge_realizations: bool = False

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

    @model_validator(mode="after")
    def _check(self) -> Self:
        check_enum_values(self.perturbation_types, PerturbationType)
        check_enum_values(self.boundary_types, BoundaryType)
        return self
