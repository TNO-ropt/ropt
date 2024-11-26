"""Configuration class for objective functions."""

from __future__ import annotations

from typing import Self

import numpy as np
from pydantic import ConfigDict, model_validator

from ropt.config.utils import (
    ImmutableBaseModel,
    broadcast_1d_array,
    broadcast_arrays,
    normalize,
)
from ropt.config.validated_types import (  # noqa: TC001
    Array1D,
    Array1DBool,
    Array1DInt,
    UniqueNames,
)


class ObjectiveFunctionsConfig(ImmutableBaseModel):
    """The configuration class for objective functions.

    This configuration class defines objective functions configured by the
    `objectives` field in an [`EnOptConfig`][ropt.config.enopt.EnOptConfig]
    object.

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

    Info: Manual scaling and auto_scaling
        Both the `scales` values and the values obtained by auto-scaling will be
        applied. Thus, if `scales` is not supplied, auto-scaling will scale the
        objectives such that their initial values will be equal to one. Setting
        `scales` also allows for scaling to different initial values.

    The objective functions may be subject to realization filters and function
    transforms. The `realization_filters` and `function_transforms` fields
    contain indices to the realization filter or function transform objects to
    use. The objects referred to are configured in the parent
    [`EnOptConfig`][ropt.config.enopt.EnOptConfig] object.

    Info: Calculating auto-scaling values
        For auto-scaling, the initial value of the objective functions is used
        based on the assumption that it is calculated as a weighted sum from an
        ensemble of realizations, with weights specified by the `realizations`
        field in the [`EnOptConfig`][ropt.config.enopt.EnOptConfig] object. If
        the `realizations_filters` and/or `function_transforms` fields are also
        set, this assumption may not strictly hold; however, the scaling value
        will still be calculated in the same way for practical purposes.

    Attributes:
        names:               Optional names of the objective functions.
        weights:             Objective functions weights (default: 1.0).
        scales:              The scaling factors (default: 1.0).
        auto_scale:          Enable/disable auto-scaling (default: `False`).
        realization_filters: Optional realization filter indices.
        function_transforms: Optional function transform indices.
    """

    names: UniqueNames | None = None
    weights: Array1D = np.array(1.0)
    scales: Array1D = np.array(1.0, dtype=np.float64)
    auto_scale: Array1DBool = np.array(False)  # noqa: FBT003
    realization_filters: Array1DInt | None = None
    function_transforms: Array1DInt | None = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_default=True,
    )

    @model_validator(mode="after")
    def _broadcast_and_normalize(self) -> Self:
        self._mutable()
        if self.names is not None:
            size = len(self.names)
            for name in ("scales", "auto_scale", "weights"):
                setattr(self, name, broadcast_1d_array(getattr(self, name), name, size))
        else:
            self.scales, self.auto_scale, self.weights = broadcast_arrays(
                self.scales, self.auto_scale, self.weights
            )

        self.weights = normalize(self.weights)
        self._immutable()
        return self
