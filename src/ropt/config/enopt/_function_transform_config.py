"""Configuration class for function transforms."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ._enopt_base_model import EnOptBaseModel
from .constants import DEFAULT_FUNCTION_TRANSFORM_BACKEND


class FunctionTransformConfig(EnOptBaseModel):
    """Configuration class for function transforms.

    This class defines the configuration for function transforms, which are
    configured by the `function_transforms` field in an
    [`EnOptConfig`][ropt.config.enopt.EnOptConfig] object. That field contains a
    tuple of configuration objects that define which function transforms are
    available during the optimization.

    By default, the final objective and constraint functions and their gradients
    are calculated from the individual realizations by a weighted sum. Function
    transforms are optionally used to modify this calculation.

    Function transforms are performed by a function transform backend, which
    provides the methods that implement the calculation of the final function or
    gradient from the individual realizations. The `backend` field is used to
    select the backend, which may be either built-in or installed separately as
    a plugin. A backend may implement multiple algorithms, and the `method`
    field determines which one will be used. To further specify how such a
    method should function, the `options` field can be used to pass a dictionary
    of key-value pairs. The interpretation of these options depends on the
    backend and the chosen method.

    Attributes:
        backend: The name of the function transform backend (default:
            [`DEFAULT_FUNCTION_TRANSFORM_BACKEND`][ropt.config.enopt.constants.DEFAULT_FUNCTION_TRANSFORM_BACKEND])
        method:  The function transform method
        options: Options to be passed to the transform
    """

    backend: str = DEFAULT_FUNCTION_TRANSFORM_BACKEND
    method: Optional[str] = None
    options: Dict[str, Any] = {}  # noqa: RUF012
