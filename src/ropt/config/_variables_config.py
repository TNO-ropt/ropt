"""Configuration class for variables."""

from __future__ import annotations

from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ropt._utils import (
    broadcast_1d_array,
    check_enum_values,
)
from ropt.enums import BoundaryType, PerturbationType, VariableType

from ._validated_types import (  # noqa: TC001  # noqa: TC001
    Array1D,
    Array1DBool,
    Array1DInt,
    ArrayEnum,
    ItemOrTuple,
)
from .constants import (
    DEFAULT_PERTURBATION_BOUNDARY_TYPE,
    DEFAULT_PERTURBATION_MAGNITUDE,
    DEFAULT_PERTURBATION_TYPE,
    DEFAULT_SEED,
)


class VariablesConfig(BaseModel):
    r"""Configuration class for optimization variables.

    `VariablesConfig` defines optimization variable settings for an
    [`EnOptContext`][ropt.context.EnOptContext] object: bounds, types, mask, and
    perturbation settings.

    See the [Configuration guide](../usage/configuration.md#variables) for
    detailed descriptions and usage examples.

    Attributes:
        variable_count:           Number of variables.
        lower_bounds:             Lower bounds for the variables (default: $-\infty$).
        upper_bounds:             Upper bounds for the variables (default: $+\infty$).
        types:                    Optional variable types.
        mask:                     Optional boolean mask indicating free variables.
        perturbation_magnitudes:  Magnitudes of the perturbations for each variable
            (default:
            [`DEFAULT_PERTURBATION_MAGNITUDE`][ropt.config.constants.DEFAULT_PERTURBATION_MAGNITUDE]).
        perturbation_types:       Type of perturbation for each variable (see
            [`PerturbationType`][ropt.enums.PerturbationType], default:
            [`DEFAULT_PERTURBATION_TYPE`][ropt.config.constants.DEFAULT_PERTURBATION_TYPE]).
        boundary_types:           How to handle perturbations that violate boundary
            conditions (see [`BoundaryType`][ropt.enums.BoundaryType], default:
            [`DEFAULT_PERTURBATION_BOUNDARY_TYPE`][ropt.config.constants.DEFAULT_PERTURBATION_BOUNDARY_TYPE]).
        samplers:                 Indices of the samplers to use for each variable.
        seed:                     Seed for the random number generator used by the samplers.
        transforms:               Indices of the variable transforms to apply for each variable.
    """

    variable_count: int
    lower_bounds: Array1D = np.array(-np.inf)
    upper_bounds: Array1D = np.array(np.inf)
    types: ArrayEnum = np.array(VariableType.REAL)
    mask: Array1DBool = np.array(1)
    perturbation_magnitudes: Array1D = np.array(DEFAULT_PERTURBATION_MAGNITUDE)
    perturbation_types: ArrayEnum = np.array(DEFAULT_PERTURBATION_TYPE)
    boundary_types: ArrayEnum = np.array(DEFAULT_PERTURBATION_BOUNDARY_TYPE)
    samplers: Array1DInt = np.array(0)
    seed: ItemOrTuple[int] = (DEFAULT_SEED,)
    transforms: Array1DInt = np.array(-1)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_default=True,
        frozen=True,
    )

    @field_validator("types", mode="after")
    @classmethod
    def _check_variable_types(cls, value: ArrayEnum) -> ArrayEnum:
        check_enum_values(value, VariableType)
        return value

    @field_validator("perturbation_types", mode="after")
    @classmethod
    def _check_perturbation_types(cls, value: ArrayEnum) -> ArrayEnum:
        check_enum_values(value, PerturbationType)
        return value

    @field_validator("boundary_types", mode="after")
    @classmethod
    def _check_boundary_types(cls, value: ArrayEnum) -> ArrayEnum:
        check_enum_values(value, BoundaryType)
        return value

    @model_validator(mode="after")
    def _broadcast_and_transform(self) -> Self:
        dim = self.variable_count
        lower_bounds = broadcast_1d_array(self.lower_bounds, "lower_bounds", dim)
        upper_bounds = broadcast_1d_array(self.upper_bounds, "upper_bounds", dim)
        types = broadcast_1d_array(self.types, "types", dim)
        mask = broadcast_1d_array(self.mask, "mask", dim)
        perturbation_magnitudes = broadcast_1d_array(
            self.perturbation_magnitudes, "perturbation_magnitudes", dim
        )
        perturbation_types = broadcast_1d_array(
            self.perturbation_types, "perturbation_types", dim
        )
        boundary_types = broadcast_1d_array(self.boundary_types, "boundary_types", dim)
        samplers = broadcast_1d_array(self.samplers, "samplers", dim)
        transforms = broadcast_1d_array(self.transforms, "transforms", dim)

        if np.any(lower_bounds > upper_bounds):
            msg = "The lower bounds are larger than the upper bounds."
            raise ValueError(msg)

        relative = perturbation_types == PerturbationType.RELATIVE
        if not np.all(
            np.logical_and(
                np.isfinite(lower_bounds[relative]), np.isfinite(upper_bounds[relative])
            ),
        ):
            msg = "The variable bounds must be finite to use relative perturbations"
            raise ValueError(msg)

        return self.model_copy(
            update={
                "types": types,
                "lower_bounds": lower_bounds,
                "upper_bounds": upper_bounds,
                "mask": mask,
                "perturbation_magnitudes": perturbation_magnitudes,
                "perturbation_types": perturbation_types,
                "boundary_types": boundary_types,
                "samplers": samplers,
                "transforms": transforms,
            }
        )
