"""Configuration class for function estimators."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class VariableTransformConfig(BaseModel):
    """Configuration class for variable transforms.

    `VariableTransformConfig` configures a
    [`VariableTransform`][ropt.transforms.VariableTransform] plugin that
    transforms variables to the optimizer's domain. Variable transforms are
    configured as a tuple in the `variable_transforms` field of
    [`EnOptContext`][ropt.context.EnOptContext]. Variables reference a specific
    transform by its index in that tuple.

    The `method` field specifies the transform method to use for the variables.
    The `options` field allows passing a dictionary of key-value pairs to
    further configure the chosen method. The interpretation of these options
    depends on the selected method.

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
    transforms objective values. Objective transforms are configured as a tuple
    in the `objective_transforms` field of
    [`EnOptContext`][ropt.context.EnOptContext]. Objectives reference a specific
    transform by its index in that tuple.

    The `method` field specifies the transform method to use for the objectives.
    The `options` field allows passing a dictionary of key-value pairs to
    further configure the chosen method. The interpretation of these options
    depends on the selected method.

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
    plugin that transforms constraint values. Nonlinear constraint transforms are
    configured as a tuple in the `nonlinear_constraint_transforms` field of
    [`EnOptContext`][ropt.context.EnOptContext]. Constraints reference a specific
    transform by its index in that tuple.

    The `method` field specifies the transform method to use for the nonlinear
    constraints. The `options` field allows passing a dictionary of key-value
    pairs to further configure the chosen method. The interpretation of these
    options depends on the selected method.

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
