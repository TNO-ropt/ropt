"""The `EnOptConfig` configuration class."""

from __future__ import annotations

import sys
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from ropt.config.utils import immutable_array
from ropt.enums import PerturbationType

from ._function_transform_config import FunctionTransformConfig
from ._gradient_config import GradientConfig
from ._linear_constraints_config import LinearConstraintsConfig
from ._nonlinear_constraints_config import NonlinearConstraintsConfig  # noqa: TCH001
from ._objective_functions_config import ObjectiveFunctionsConfig
from ._optimizer_config import OptimizerConfig
from ._realization_filter_config import RealizationFilterConfig  # noqa: TCH001
from ._realizations_config import RealizationsConfig
from ._sampler_config import SamplerConfig
from ._variables_config import VariablesConfig  # noqa: TCH001

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


class EnOptConfig(BaseModel):
    """The primary configuration class for a single optimization step.

    The fields of the `EnOptConfig` class are nested configuration classes that
    specify specific aspects of a single optimization run.

    Upon initialization and validation of an `EnOptConfig` object, the contents
    of some of these fields may be modified, depending on the content of other
    fields. A good example is the scaling of variables: The `offsets` and
    `scales` field in the `variables` field of an `EnOptConfig` object define a
    linear transformation of the variables. Upon initialization, values of the
    `linear_constraints` field will be transformed accordingly so that the
    constraints remain valid for the transformed variables.

    The `realization_filters`, `function_transforms`, and `samplers` fields are
    defined as tuples of configurations for realization filter, function
    transform, and sampler objects, respectively. Other configuration fields
    will refer to these objects by their index into these tuples. For example,
    the `gradient` field is implemented by the
    [`GradientConfig`][ropt.config.enopt.GradientConfig] class, which contains a
    `samplers` field that is an array of indices, indicating for each variable
    which sampler should be used.

    The original values of all fields used to create the object will be stored
    internally and are available via the `original_inputs` field.

    Info:
        Many of these nested classes contain fields that are
        [`numpy`](https://np.org) arrays of values. In general, these arrays
        must have a given size defined by the configured property or a size of
        one. For instance, the `variables` field must be an object of the
        [`VariablesConfig`][ropt.config.enopt.VariablesConfig] class, which
        contains information about the variables to be optimized. This includes
        such properties as initial values, bounds, and so on, which are defined
        as `numpy` arrays. The size of these arrays must be either equal to the
        number of variables or equal to one, in which case that single value is
        used for all variables.

    Attributes:
        variables:             Configuration of the variables.
        objective_functions:   Configuration of the objective functions.
        linear_constraints:    Configuration of linear constraints.
        nonlinear_constraints: Configuration of non-linear constraints.
        realizations:          Configuration of the realizations.
        optimizer:             Configuration of the optimizer.
        gradient:              Configuration for gradient calculations.
        realization_filters:   Configuration of realization filters.
        function_transforms:   Configuration of function transforms.
        samplers:              Configuration of samplers.
        original_inputs:       The original input to the constructor.
    """

    variables: VariablesConfig
    objective_functions: ObjectiveFunctionsConfig = ObjectiveFunctionsConfig()
    linear_constraints: Optional[LinearConstraintsConfig] = None
    nonlinear_constraints: Optional[NonlinearConstraintsConfig] = None
    realizations: RealizationsConfig = RealizationsConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    gradient: GradientConfig = GradientConfig()
    realization_filters: Tuple[RealizationFilterConfig, ...] = ()
    function_transforms: Tuple[FunctionTransformConfig, ...] = (
        FunctionTransformConfig(),
    )
    samplers: Tuple[SamplerConfig, ...] = (SamplerConfig(),)
    original_inputs: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
    )

    @model_validator(mode="after")
    def _linear_constraints(self) -> Self:
        if self.linear_constraints is not None:
            variable_count = self.variables.initial_values.size
            if (
                self.linear_constraints.coefficients.shape[0] > 0
                and self.linear_constraints.coefficients.shape[1] != variable_count
            ):
                msg = f"the coefficients matrix should have {variable_count} columns"
                raise ValueError(msg)

            # Correct the linear system of input constraints for scaling:
            offsets = self.variables.offsets
            scales = self.variables.scales
            if offsets is not None or scales is not None:
                coefficients = self.linear_constraints.coefficients
                rhs_values = self.linear_constraints.rhs_values
                if offsets is not None:
                    rhs_values = rhs_values - np.matmul(coefficients, offsets)
                if scales is not None:
                    coefficients = coefficients * scales
                values = self.linear_constraints.model_dump(round_trip=True)
                values.update(
                    coefficients=coefficients,
                    rhs_values=rhs_values,
                    types=self.linear_constraints.types,
                )
                self.linear_constraints = LinearConstraintsConfig.model_construct(
                    **values,
                )
        return self

    @model_validator(mode="after")
    def _gradient(self) -> Self:
        variables = self.variables
        variable_count = variables.initial_values.size
        magnitudes = self.gradient.perturbation_magnitudes
        boundary_types = self.gradient.boundary_types
        types = self.gradient.perturbation_types

        try:
            magnitudes = np.broadcast_to(magnitudes, (variable_count,))
        except ValueError as err:
            msg = (
                "the perturbation magnitudes cannot be broadcasted "
                f"to a length of {variable_count}"
            )
            raise ValueError(msg) from err

        if boundary_types.size == 1:
            boundary_types = np.broadcast_to(
                immutable_array(boundary_types),
                (variable_count,),
            )
        elif boundary_types.size == variable_count:
            boundary_types = immutable_array(boundary_types)
        else:
            msg = f"perturbation boundary_types must have {variable_count} items"
            raise ValueError(msg)

        if types.size == 1:
            types = np.broadcast_to(immutable_array(types), (variable_count,))
        elif types.size == variable_count:
            types = immutable_array(types)
        else:
            msg = f"perturbation types must have {variable_count} items"
            raise ValueError(msg)

        relative = types == PerturbationType.RELATIVE
        if not np.all(
            np.logical_and(
                np.isfinite(variables.lower_bounds[relative]),
                np.isfinite(variables.lower_bounds[relative]),
            ),
        ):
            msg = "The variable bounds must be finite to use relative perturbations"
            raise ValueError(msg)
        magnitudes = np.where(
            relative,
            (variables.upper_bounds - variables.lower_bounds) * magnitudes,
            magnitudes,
        )

        if variables.scales is not None:
            scaled = types == PerturbationType.SCALED
            magnitudes = np.where(scaled, magnitudes / variables.scales, magnitudes)

        self.gradient.perturbation_magnitudes = magnitudes
        self.gradient.boundary_types = boundary_types
        self.gradient.perturbation_types = types

        return self

    @model_validator(mode="before")
    @classmethod
    def _add_original_data(cls, data: Any) -> Any:  # noqa: ANN401
        if isinstance(data, dict):
            newdata = deepcopy(data)
            newdata["original_inputs"] = deepcopy(data)
            return newdata
        msg = "EnOptConfig objects must be constructed from dicts"
        raise ValueError(msg)

    @model_validator(mode="wrap")  # type: ignore[arg-type]
    def _pass_enopt_config_unchanged(self, handler: Any) -> Any:  # noqa: ANN401
        if isinstance(self, EnOptConfig):
            return self
        return handler(self)
