"""The `EnOptConfig` configuration class."""

from __future__ import annotations

from typing import Any, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationInfo, model_validator

from ropt.enums import PerturbationType

from ._function_estimator_config import FunctionEstimatorConfig
from ._gradient_config import GradientConfig
from ._linear_constraints_config import LinearConstraintsConfig  # noqa: TC001
from ._nonlinear_constraints_config import NonlinearConstraintsConfig  # noqa: TC001
from ._objective_functions_config import ObjectiveFunctionsConfig
from ._optimizer_config import OptimizerConfig
from ._realization_filter_config import RealizationFilterConfig  # noqa: TC001
from ._realizations_config import RealizationsConfig
from ._sampler_config import SamplerConfig
from ._variables_config import VariablesConfig  # noqa: TC001


class EnOptConfig(BaseModel):
    """The primary configuration class for an optimization step.

    `EnOptConfig` orchestrates the configuration of an entire optimization
    workflow. It contains nested configuration classes that define specific
    aspects of the optimization, such as variables, objectives, constraints,
    realizations, and the optimizer itself.

    `realization_filters`, `function_estimators`, and `samplers` are configured
    as tuples. Other configuration fields reference these objects by their index
    within the tuples. For example,
    [`GradientConfig`][ropt.config.enopt.GradientConfig] uses a `samplers`
    field, which is an array of indices specifying the sampler to use for each
    variable.

    Info:
        Many nested configuration classes use `numpy` arrays. These arrays
        typically have a size determined by a configured property (e.g., the
        number of variables) or a size of one. In the latter case, the single
        value is broadcasted to all relevant elements. For example,
        [`VariablesConfig`][ropt.config.enopt.VariablesConfig] defines
        properties like initial values and bounds as `numpy` arrays, which must
        either match the number of variables or have a size of one.

    Attributes:
        variables:             Configuration for the optimization variables.
        objectives:            Configuration for the objective functions.
        linear_constraints:    Configuration for linear constraints.
        nonlinear_constraints: Configuration for non-linear constraints.
        realizations:          Configuration for the realizations.
        optimizer:             Configuration for the optimization algorithm.
        gradient:              Configuration for gradient calculations.
        realization_filters:   Configuration for realization filters.
        function_estimators:   Configuration for function estimators.
        samplers:              Configuration for samplers.
    """

    variables: VariablesConfig
    objectives: ObjectiveFunctionsConfig = ObjectiveFunctionsConfig()
    linear_constraints: LinearConstraintsConfig | None = None
    nonlinear_constraints: NonlinearConstraintsConfig | None = None
    realizations: RealizationsConfig = RealizationsConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    gradient: GradientConfig = GradientConfig()
    realization_filters: tuple[RealizationFilterConfig, ...] = ()
    function_estimators: tuple[FunctionEstimatorConfig, ...] = (
        FunctionEstimatorConfig(),
    )
    samplers: tuple[SamplerConfig, ...] = (SamplerConfig(),)

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
    )

    @model_validator(mode="after")
    def _check_linear_constraints(self) -> Self:
        if self.linear_constraints is not None:
            variable_count = self.variables.initial_values.size
            if (
                self.linear_constraints.coefficients.shape[0] > 0
                and self.linear_constraints.coefficients.shape[1] != variable_count
            ):
                msg = f"the coefficients matrix should have {variable_count} columns"
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_magnitudes(self) -> Self:
        size = self.gradient.perturbation_magnitudes.size
        if size not in (1, self.variables.initial_values.size):
            msg = (
                "the perturbation magnitudes cannot be broadcasted "
                f"to a length of {self.variables.initial_values.size}"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_boundary_types(self) -> Self:
        size = self.gradient.boundary_types.size
        if size not in (1, self.variables.initial_values.size):
            msg = (
                "perturbation boundary_types must have "
                f"{self.variables.initial_values.size} items, or a single item"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_perturbation_types(self) -> Self:
        size = self.gradient.perturbation_types.size
        if size not in (1, self.variables.initial_values.size):
            msg = (
                "perturbation types must have "
                f"{self.variables.initial_values.size} items, or a single item"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_relative_magnitudes(self) -> Self:
        types = np.broadcast_to(
            self.gradient.perturbation_types, (self.variables.initial_values.size,)
        )
        relative = types == PerturbationType.RELATIVE
        if not np.all(
            np.logical_and(
                np.isfinite(self.variables.lower_bounds[relative]),
                np.isfinite(self.variables.upper_bounds[relative]),
            ),
        ):
            msg = "The variable bounds must be finite to use relative perturbations"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _fix_linear_constraints(self, info: ValidationInfo) -> Self:
        if self.linear_constraints is not None:
            self.linear_constraints = self.linear_constraints.apply_transformation(
                info.context
            )
        return self

    @model_validator(mode="after")
    def _gradient(self, info: ValidationInfo) -> Self:
        self.gradient = self.gradient.fix_perturbations(self.variables, info.context)
        return self

    @model_validator(mode="wrap")  # type: ignore[arg-type]
    def _pass_enopt_config_unchanged(self, handler: Any) -> Any:  # noqa: ANN401
        if isinstance(self, EnOptConfig):
            return self
        return handler(self)
