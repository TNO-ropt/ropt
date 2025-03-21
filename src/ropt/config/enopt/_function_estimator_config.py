"""Configuration class for function estimators."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class FunctionEstimatorConfig(BaseModel):
    """Configuration class for function estimators.

    This class, `FunctionEstimatorConfig`, defines the configuration for
    function estimators used in an
    [`EnOptConfig`][ropt.config.enopt.EnOptConfig] object. Function estimators
    are configured as a tuple in the `function_estimators` field of the
    `EnOptConfig`, defining the available estimators for the optimization.

    By default, objective and constraint functions, as well as their gradients,
    are calculated from individual realizations using a weighted sum. Function
    estimators provide a way to modify this default calculation.

    The `method` field specifies the function estimator method to use for
    combining the individual realizations. The `options` field allows passing a
    dictionary of key-value pairs to further configure the chosen method. The
    interpretation of these options depends on the selected method.

    Attributes:
        method:  Name of the function estimator method.
        options: Dictionary of options for the function estimator.
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
