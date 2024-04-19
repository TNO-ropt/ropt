"""Configuration class for realization filters."""

from __future__ import annotations

from typing import Any, Dict

from ._enopt_base_model import EnOptBaseModel
from .constants import DEFAULT_REALIZATION_FILTER_BACKEND


class RealizationFilterConfig(EnOptBaseModel):
    """The configuration class for realization filters.

    This class defines the configuration for realization filters, which are
    configured by the `realization_filters` field in an
    [`EnOptConfig`][ropt.config.enopt.EnOptConfig] object. This field contains a
    tuple of configuration objects that define which realization filters are
    available during optimization.

    By default, the final objective and constraint functions and their gradients
    are calculated as a weighted function from all realizations. Realization
    filters are optionally used to change the weights of the individual
    realizations. For instance, this can be used to determine which subset of
    realizations should be used in calculating the final objective and
    constraint functions and their gradients by setting some weights to zero.

    Filtering is performed by a realization filter backend, which provides the
    methods that can be used to adjust the weights of the individual
    realizations. The `backend` field is used to select the backend, which may
    be either built-in or installed separately as a plugin. A backend may
    implement multiple algorithms, and the `method` field determines which one
    will be used. To further specify how such a method should function, the
    `options` field can be used to pass a dictionary of key-value pairs. The
    interpretation of these options depends on the backend and the chosen
    method.

    Attributes:
        backend: The name of the realization filter backend (default:
            [`DEFAULT_REALIZATION_FILTER_BACKEND`][ropt.config.enopt.constants.DEFAULT_REALIZATION_FILTER_BACKEND])
        method:  The realization filter method
        options: Options to be passed to the filter
    """

    backend: str = DEFAULT_REALIZATION_FILTER_BACKEND
    method: str
    options: Dict[str, Any] = {}  # noqa: RUF012
