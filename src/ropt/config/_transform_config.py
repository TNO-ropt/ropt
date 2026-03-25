"""Configuration class for function estimators."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class VariableTransformConfig(BaseModel):
    """Configuration class for variable transforms.

    This class, `VariableTransformConfig`, defines the configuration for
    variable transforms. Variable transforms are generally configured as a tuple
    of `VariableTransformConfig` objects in a configuration class of an
    optimization step. For instance, `variable_transforms` field of the
    `EnOptConfig` defines the available transforms for the optimization.


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

    This class, `ObjectiveTransformConfig`, defines the configuration for
    objective transforms. Objective transforms are generally configured as a
    tuple of `ObjectiveTransformConfig` objects in a configuration class of an
    optimization step. For instance, `objective_transforms` field of the
    `EnOptConfig` defines the available transforms for the optimization.

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

    This class, `NonlinearConstraintTransformConfig`, defines the configuration
    for nonlinear constraint transforms. Nonlinear constraint transforms are
    generally configured as a tuple of `NonlinearConstraintTransformConfig`
    objects in a configuration class of an optimization step. For instance,
    `nonlinear_constraint_transforms` field of the `EnOptConfig` defines the
    available transforms for the optimization.

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
