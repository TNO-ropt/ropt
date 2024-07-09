"""Configuration class for objective functions."""

from __future__ import annotations

from typing import Optional

import numpy as np
from pydantic import model_validator

from ropt.config.utils import (
    Array1D,
    Array1DBool,
    Array1DInt,
    UniqueNames,
    broadcast_1d_array,
    broadcast_arrays,
    normalize,
)

from ._enopt_base_model import EnOptBaseModel


class ObjectiveFunctionsConfig(EnOptBaseModel):
    """The configuration class for objective functions.

    This configuration class defines objective functions configured by the
    `objective_functions` field in an
    [`EnOptConfig`][ropt.config.enopt.EnOptConfig] object.

    `ropt` supports optimization over multiple objectives, which are summed after
    weighting with values passed via the `weights` field. This field is a
    `numpy` array, with a length equal to the number of objective functions. Its
    values will be normalized to have a sum equal to 1. For example, when
    `weights` is set to `[1, 1]`, the stored values will be `[0.5, 0.5]`.

    The `names` field is optional. If given, the number of objective functions
    is set equal to its length. The `weights` array will then be broadcasted to
    the number of objective values. For example, if `names = ["f1", "f2"]` and
    `weights = 1.0`, the optimizer assumes two objective functions weighted by
    `[0.5, 0.5]`. If `names` is not set, the number of objectives is determined
    by the length of `weights`.

    The `scales` field contains scaling values for the objectives. These values
    are used to scale the objective function values to a desired order of
    magnitude. Each time new objective function values are obtained during
    optimization, they are divided by these values. The `auto_scale` field can
    be used to direct the optimizer to obtain an additional scaling by
    multiplying the values of `scales` by the values of the objective functions
    at the start of the optimization. Both the `scales` and `auto_scale` arrays
    will be broadcasted to have a size equal to the number of objective
    functions.

    Info:
        Both the `scales` values and the values obtained by auto-scaling will be
        applied. Thus, if `scales` is not supplied, auto-scaling will scale the
        objectives such that their initial values will be equal to one. Setting
        `scales` also allows for scaling to different initial values.

    The objective functions may be subject to realization filters and function
    transforms. The `realization_filters` and `function_transforms` fields
    contain indices to the realization filter or function transform objects to
    use. The objects referred to are configured in the parent
    [`EnOptConfig`][ropt.config.enopt.EnOptConfig] object.

    Attributes:
        names:               Optional names of the objective functions
        weights:             Objective functions weights (default: 1.0)
        scales:              The scaling factors (default: 1.0)
        auto_scale:          Enable/disable auto-scaling (default: `False`)
        realization_filters: Optional realization filter indices
        function_transforms: Optional function transform indices
    """

    names: Optional[UniqueNames] = None
    weights: Array1D = np.array(1.0)
    scales: Array1D = np.array(1.0, dtype=np.float64)
    auto_scale: Array1DBool = np.array(False)  # noqa: FBT003
    realization_filters: Optional[Array1DInt] = None
    function_transforms: Optional[Array1DInt] = None

    @model_validator(mode="after")
    def _broadcast_and_normalize(self) -> ObjectiveFunctionsConfig:
        if self.names is not None:
            size = len(self.names)
            for name in ("scales", "auto_scale", "weights"):
                setattr(self, name, broadcast_1d_array(getattr(self, name), name, size))
        else:
            self.scales, self.auto_scale, self.weights = broadcast_arrays(
                self.scales, self.auto_scale, self.weights
            )

        self.weights = normalize(self.weights)
        return self
