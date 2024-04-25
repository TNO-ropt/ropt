"""Configuration class for realizations."""

from __future__ import annotations

from typing import Optional

import numpy as np
from pydantic import NonNegativeInt, model_validator

from ropt.config.utils import (
    Array1D,
    UniqueNames,
    broadcast_1d_array,
    normalize,
)

from ._enopt_base_model import EnOptBaseModel


class RealizationsConfig(EnOptBaseModel):
    """The configuration class for realizations.

    This class defines realizations configured by the `realizations` field in an
    [`EnOptConfig`][ropt.config.enopt.EnOptConfig] object.

    To optimize an ensemble of functions, a set of realizations is defined. When
    the optimizer requests a function value or a gradient, the functions and
    gradients are calculated for each realization and combined into a single
    function or gradient. Usually, this will be a (weighted) sum, but other ways
    of combining realizations are possible (see [Realization
    filters](realization_filters.md) and [Function
    transforms](function_transforms.md) for more details).

    The `weights` field is a `numpy` array, with a length equal to the number of
    realizations. Its values will be normalized to have a sum equal to 1. For
    example, when `weights` is set to `[1, 1]`, the stored values will be `[0.5,
    0.5]`.

    The `names` field is optional. If given, the number of realizations is set
    equal to its length. The `weights` array will then be broadcasted to the
    number of objective values. For example, if `names = ["r1", "r2"]` and
    `weights = 1.0`, the optimizer assumes two realizations weighted by `[0.5,
    0.5]`. If `names` is not set, the number of realizations is determined by
    the length of `weights`.

    If during the calculation of the function values for each realization one or
    more values are missing, for instance due to failure of a complex
    simulation, the total function and gradient values can still be calculated
    by leaving the missing values out. However, this may be undesirable, or
    there may be a hard minimum to the amount of values that is needed. The
    `realization_min_success` field can be set to the minimum number of
    successful realizations. By default, it is set equal to the number of
    realizations, i.e., there are no missing values allowed by default.

    Note:
        The value of `realization_min_success` can be set to zero. Some
        optimizers can handle this and will proceed with the optimization even
        if all realizations fail. However, most optimizers cannot handle this
        and will behave as if the value is set to one.

    Attributes:
        names:                   Optional names of the realizations
        weights:                 The weights of the realizations (default: 1)
        realization_min_success: The minimum number of successful realizations
                                 (default: equal to the number of realizations)
    """

    names: Optional[UniqueNames] = None
    weights: Array1D = np.array(1.0)
    realization_min_success: Optional[NonNegativeInt] = None

    @model_validator(mode="after")
    def _broadcast_normalize_and_check(self) -> RealizationsConfig:
        if self.names:
            size = len(self.names)
            self.weights = broadcast_1d_array(self.weights, "weights", size)
        else:
            self.weights = np.array(self.weights, copy=False, ndmin=1)
            if self.weights.ndim != 1:
                msg = "input arrays cannot be broadcasted to an 1d array"
                raise ValueError(msg)

        self.weights = normalize(self.weights)

        if (
            self.realization_min_success is None
            or self.realization_min_success > self.weights.size
        ):
            self.realization_min_success = self.weights.size

        return self
