"""Configuration class for function transforms."""

from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, ConfigDict


class FunctionTransformConfig(BaseModel):
    """Configuration class for function transforms.

    This class defines the configuration for function transforms, which are
    configured by the `function_transforms` field in an
    [`EnOptConfig`][ropt.config.enopt.EnOptConfig] object. That field contains a
    tuple of configuration objects that define which function transforms are
    available during the optimization.

    By default, the final objective and constraint functions and their gradients
    are calculated from the individual realizations by a weighted sum. Function
    transforms are optionally used to modify this calculation.

    The `method` field determines which method will be used to implement the
    calculation of the final function or gradient from the individual
    realizations. To further specify how such a method should function, the
    `options` field can be used to pass a dictionary of key-value pairs. The
    interpretation of these options depends on the chosen method.

    Attributes:
        method:  The function transform method
        options: Options to be passed to the transform
    """

    method: str = "default/default"
    options: Dict[str, Any] = {}

    model_config = ConfigDict(
        extra="forbid",
        str_min_length=1,
        str_strip_whitespace=True,
        validate_default=True,
    )
