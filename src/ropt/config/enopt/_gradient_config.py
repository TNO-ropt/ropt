"""Configuration class for gradients."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Optional

import numpy as np
from pydantic import ConfigDict, PositiveInt, model_validator

from ropt.config.utils import ImmutableBaseModel, check_enum_values, immutable_array
from ropt.config.validated_types import (  # noqa: TCH001
    Array1D,
    Array1DInt,
    ArrayEnum,
    ItemOrTuple,
)
from ropt.enums import BoundaryType, PerturbationType

from .constants import (
    DEFAULT_NUMBER_OF_PERTURBATIONS,
    DEFAULT_PERTURBATION_BOUNDARY_TYPE,
    DEFAULT_PERTURBATION_MAGNITUDE,
    DEFAULT_PERTURBATION_TYPE,
    DEFAULT_SEED,
)

if TYPE_CHECKING:
    from ropt.config.enopt import VariablesConfig

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


class GradientConfig(ImmutableBaseModel):
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
    perturbation values for that variable. To support samplers that need random number,
    a random number generator object is created

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
            [`DEFAULT_NUMBER_OF_PERTURBATIONS`][ropt.config.enopt.constants.DEFAULT_NUMBER_OF_PERTURBATIONS]).
        perturbation_min_success: The minimum number of successful function
                                  evaluations for perturbed variables (default:
                                  equal to the number of perturbations).
        perturbation_magnitudes:  The magnitudes of the perturbations for each variable
                                  (default:
            [`DEFAULT_PERTURBATION_MAGNITUDE`][ropt.config.enopt.constants.DEFAULT_PERTURBATION_MAGNITUDE]).
        perturbation_types:       The type of perturbation for each variable
                                  ([`PerturbationType`][ropt.enums.PerturbationType],
                                  default:
            [`DEFAULT_PERTURBATION_TYPE`][ropt.config.enopt.constants.DEFAULT_PERTURBATION_TYPE]).
        boundary_types:           How perturbations that violate boundary conditions
                                  are treated (see [`BoundaryType`][ropt.enums.BoundaryType]),
                                  default:
            [`DEFAULT_PERTURBATION_BOUNDARY_TYPE`][ropt.config.enopt.constants.DEFAULT_PERTURBATION_BOUNDARY_TYPE]).
        samplers:                 The index of the sampler to use for each variable.
        seed:                     The seed for the random number generator passed to each sampler.
        merge_realizations:       If all realizations should be merged for the final
                                  gradient calculation (default: `False`).

    Note: The seed for samples
        The seed controls consistency in results across repeated runs within the
        same plan, as long as the seed remains unchanged. To obtain unique
        results for each optimization run, the seed should be modified. The
        `numpy` manual suggests converting the seed to a tuple and pre-pending
        one or more unique integers.

        A suitable approach is to use the unique plan ID of the optimizer as the
        pre-pended value, which ensures reproducibility across nested and
        parallel plan evaluations.
    """

    number_of_perturbations: PositiveInt = DEFAULT_NUMBER_OF_PERTURBATIONS
    perturbation_min_success: Optional[PositiveInt] = None
    perturbation_magnitudes: Array1D = np.array(DEFAULT_PERTURBATION_MAGNITUDE)
    perturbation_types: ArrayEnum = np.array(DEFAULT_PERTURBATION_TYPE)
    boundary_types: ArrayEnum = np.array(DEFAULT_PERTURBATION_BOUNDARY_TYPE)
    samplers: Optional[Array1DInt] = None
    seed: ItemOrTuple[int] = (DEFAULT_SEED,)
    merge_realizations: bool = False

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_default=True,
    )

    @model_validator(mode="after")
    def _check_perturbation_min_success(self) -> Self:
        self._mutable()
        if (
            self.perturbation_min_success is None
            or self.perturbation_min_success > self.number_of_perturbations
        ):
            self.perturbation_min_success = self.number_of_perturbations
        self._immutable()
        return self

    @model_validator(mode="after")
    def _check(self) -> Self:
        check_enum_values(self.perturbation_types, PerturbationType)
        check_enum_values(self.boundary_types, BoundaryType)
        return self

    def fix_perturbations(self, variables: VariablesConfig) -> GradientConfig:
        """Adjust the gradient perturbation configuration.

        This method modifies the gradient's perturbation settings to account for
        variable bounds and scaling factors, as defined in the `variables`
        configuration. If bounds are set on the variables or if variable scaling
        is applied, the perturbations in the gradient configuration may need
        adjustment to reflect these constraints. This method returns an updated
        copy of the gradient configuration with the necessary modifications.

        Args:
            variables: The configuration of variables.

        Returns:
            A modified gradient configuration with applied bounds and scaling.
        """
        variable_count = variables.initial_values.size
        magnitudes = self.perturbation_magnitudes
        boundary_types = self.boundary_types
        types = self.perturbation_types

        try:
            magnitudes = np.broadcast_to(magnitudes, (variable_count,))
        except ValueError as err:
            msg = (
                "the perturbation magnitudes cannot be broadcasted "
                f"to a length of {variable_count}"
            )
            raise ValueError(msg) from err

        if boundary_types.size == 1:
            boundary_types = np.broadcast_to(
                immutable_array(boundary_types),
                (variable_count,),
            )
        elif boundary_types.size == variable_count:
            boundary_types = immutable_array(boundary_types)
        else:
            msg = f"perturbation boundary_types must have {variable_count} items"
            raise ValueError(msg)

        if types.size == 1:
            types = np.broadcast_to(immutable_array(types), (variable_count,))
        elif types.size == variable_count:
            types = immutable_array(types)
        else:
            msg = f"perturbation types must have {variable_count} items"
            raise ValueError(msg)

        relative = types == PerturbationType.RELATIVE
        if not np.all(
            np.logical_and(
                np.isfinite(variables.lower_bounds[relative]),
                np.isfinite(variables.lower_bounds[relative]),
            ),
        ):
            msg = "The variable bounds must be finite to use relative perturbations"
            raise ValueError(msg)
        magnitudes = np.where(
            relative,
            (variables.upper_bounds - variables.lower_bounds) * magnitudes,
            magnitudes,
        )

        if variables.scales is not None:
            scaled = types == PerturbationType.SCALED
            magnitudes = np.where(scaled, magnitudes / variables.scales, magnitudes)

        return self.model_copy(
            update={
                "perturbation_magnitudes": magnitudes,
                "boundary_types": boundary_types,
                "perturbation_types": types,
            }
        )
