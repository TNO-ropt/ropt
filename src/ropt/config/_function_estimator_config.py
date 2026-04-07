"""Configuration class for function estimators."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class FunctionEstimatorConfig(BaseModel):
    """Configuration class for function estimators.

    `FunctionEstimatorConfig` configures a function estimator plugin, which
    controls how objective and constraint function values (and their gradients)
    are combined across realizations. By default, a weighted sum over
    realizations is used; function estimators allow replacing that with a
    different combination method.

    The `method` field selects the estimator using a `"plugin/method"` string
    (e.g. `"default/default"`). The `options` field passes additional
    configuration to the selected method.

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
