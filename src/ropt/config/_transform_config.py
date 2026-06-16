"""Configuration class for function estimators."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class VariableTransformConfig(BaseModel):
    """Configuration class for variable transforms.

    `VariableTransformConfig` configures a
    [`VariableTransform`][ropt.transforms.VariableTransform] plugin that
    transforms variables to the optimizer's domain.

    See the [Configuration guide](../usage/configuration.md#transforms) for
    detailed descriptions and usage examples.

    Attributes:
        method:  Name of the variable transform method.
        options: Dictionary of options for the variable transform method.
    """

    method: str = "default/default"
    options: dict[str, Any] = {}

    model_config = ConfigDict(
        extra="forbid",
        str_min_length=1,
        str_strip_whitespace=True,
        validate_default=True,
        frozen=True,
    )


class ObjectiveTransformConfig(BaseModel):
    """Configuration class for objective transforms.

    `ObjectiveTransformConfig` configures an
    [`ObjectiveTransform`][ropt.transforms.ObjectiveTransform] plugin that
    transforms objective values.

    See the [Configuration guide](../usage/configuration.md#transforms) for
    detailed descriptions and usage examples.

    Attributes:
        method:  Name of the objective transform method.
        options: Dictionary of options for the objective transform method.
    """

    method: str = "default/default"
    options: dict[str, Any] = {}

    model_config = ConfigDict(
        extra="forbid",
        str_min_length=1,
        str_strip_whitespace=True,
        validate_default=True,
        frozen=True,
    )


class NonlinearConstraintTransformConfig(BaseModel):
    """Configuration class for nonlinear constraint transforms.

    `NonlinearConstraintTransformConfig` configures a
    [`NonlinearConstraintTransform`][ropt.transforms.NonlinearConstraintTransform]
    plugin that transforms constraint values.

    See the [Configuration guide](../usage/configuration.md#transforms) for
    detailed descriptions and usage examples.

    Attributes:
        method:  Name of the nonlinear constraint transform method.
        options: Dictionary of options for the nonlinear constraint transform method.
    """

    method: str = "default/default"
    options: dict[str, Any] = {}

    model_config = ConfigDict(
        extra="forbid",
        str_min_length=1,
        str_strip_whitespace=True,
        validate_default=True,
        frozen=True,
    )
