"""Configuration class for function estimators."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class FunctionEstimatorConfig(BaseModel):
    """Configuration class for function estimators.

    This class defines the configuration for function estimators, which are
    configured by the `function_estimators` field in an
    [`EnOptConfig`][ropt.config.enopt.EnOptConfig] object. That field contains a
    tuple of configuration objects that define which function estimators are
    available during the optimization.

    By default, the final objective and constraint functions and their gradients
    are calculated from the individual realizations by a weighted sum. Function
    estimators are optionally used to modify this calculation.

    The `method` field determines which method will be used to implement the
    calculation of the final function or gradient from the individual
    realizations. To further specify how such a method should function, the
    `options` field can be used to pass a dictionary of key-value pairs. The
    interpretation of these options depends on the chosen method.

    Attributes:
        method:  The function estimator method.
        options: Options to be passed to the estimator.
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
