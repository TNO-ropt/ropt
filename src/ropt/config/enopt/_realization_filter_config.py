"""Configuration class for realization filters."""

from __future__ import annotations

from typing import Any, Dict

from ._enopt_base_model import EnOptBaseModel


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

    The `method` field determines which method will be used to adjust the
    weights of the individual realizations. To further specify how such a method
    should function, the `options` field can be used to pass a dictionary of
    key-value pairs. The interpretation of these options depends on the| chosen
    method.

    Attributes:
        method:  The realization filter method
        options: Options to be passed to the filter
    """

    method: str
    options: Dict[str, Any] = {}  # noqa: RUF012
