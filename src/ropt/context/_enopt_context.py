"""The `EnOptContext` configuration class."""

from __future__ import annotations

import threading
from typing import Any, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator

from ropt._utils import immutable_array
from ropt.config import (
    FunctionEstimatorConfig,
    GradientConfig,
    LinearConstraintsConfig,
    NonlinearConstraintsConfig,
    ObjectiveFunctionsConfig,
    OptimizerConfig,
    RealizationsConfig,
    SamplerConfig,
    VariablesConfig,
)
from ropt.enums import PerturbationType
from ropt.plugins.manager import get_plugin

from ._validated_types import (  # noqa: TC001
    BackendInstance,
    FunctionEstimatorInstance,
    NonlinearConstraintTransformInstance,
    ObjectiveTransformInstance,
    RealizationFilterInstance,
    SamplerInstance,
    VariableTransformInstance,
)

_global_lock = threading.Lock()


class EnOptContext(BaseModel):
    """The primary context object for a single optimization run.

    `EnOptContext` holds all information needed to run an ensemble-based
    optimization: variables, objectives, constraints, realizations, gradient
    settings, samplers, filters, and the optimizer/backend. It is constructed
    from plain Python dicts or config objects and validated on creation.

    **Index-based sharing**

    All tuple-based plugin fields (`realization_filters`, `function_estimators`,
    `samplers`, `variable_transforms`, `objective_transforms`, and
    `nonlinear_constraint_transforms`) are referenced by index from other config
    fields. For example, the `samplers` field of
    [`VariablesConfig`][ropt.config.VariablesConfig] is an integer array whose
    values index into the `samplers` tuple — use all zeros when a single sampler
    is shared across all variables, or distinct indices when different samplers
    are needed per variable. The same pattern applies to transform indices in
    [`VariablesConfig`][ropt.config.VariablesConfig],
    [`ObjectiveFunctionsConfig`][ropt.config.ObjectiveFunctionsConfig], and
    [`NonlinearConstraintsConfig`][ropt.config.NonlinearConstraintsConfig].

    **Optional names**

    The `names` attribute maps axis types (see [`AxisName`][ropt.enums.AxisName])
    to ordered sequences of labels for variables, objectives, and constraints.
    It is not required for the optimization itself, but when present it is used
    to produce labelled multi-index results in exported data frames.

    **Plugin instances**

    The `backend` field and all tuple-based plugin fields (`realization_filters`,
    `function_estimators`, `samplers`, `variable_transforms`,
    `objective_transforms`, and `nonlinear_constraint_transforms`) store plugin
    instances. Instead of constructing instances manually, these fields can be
    initialized with a configuration object or a plain dict of settings — Pydantic
    will resolve and instantiate the appropriate plugin automatically. Each config
    class has a `method` field that selects the plugin implementation. The
    configuration classes are defined in the [`ropt.config`][ropt.config]
    sub-package.

    **Broadcasting**

    Many nested config classes represent per-variable or per-objective
    properties (e.g., bounds, perturbation magnitudes) as `numpy` arrays. A
    size-1 array is broadcast to all elements; otherwise the array length must
    match the count of the corresponding entities.

    Warning:
        `EnOptContext` objects are immutable after construction. Do not attempt
        to serialize and round-trip them (e.g., to/from JSON): `numpy` arrays
        and plugin instances cannot survive a round-trip faithfully. Persist the
        raw input dicts instead.

    Attributes:
        variables:                       Variable settings.
        objectives:                      Objective function settings.
        linear_constraints:              Optional linear constraint settings.
        nonlinear_constraints:           Optional nonlinear constraint settings.
        realizations:                    Ensemble realization settings.
        optimizer:                       Optimizer settings.
        backend:                         Backend plugin instance used for function evaluations.
        gradient:                        Gradient estimation settings.
        realization_filters:             Tuple of realization filter plugin instances.
        function_estimators:             Tuple of function estimator plugin instances.
        samplers:                        Tuple of sampler plugin instances.
        variable_transforms:             Tuple of variable transform plugin instances.
        objective_transforms:            Tuple of objective transform plugin instances.
        nonlinear_constraint_transforms: Tuple of nonlinear constraint transform plugin instances.
        names:                           Optional mapping of axis names to label sequences.
    """

    variables: VariablesConfig
    objectives: ObjectiveFunctionsConfig = ObjectiveFunctionsConfig.model_validate({})
    linear_constraints: LinearConstraintsConfig | None = None
    nonlinear_constraints: NonlinearConstraintsConfig | None = None
    realizations: RealizationsConfig = RealizationsConfig.model_validate({})
    optimizer: OptimizerConfig = OptimizerConfig.model_validate({})
    backend: BackendInstance = {}  # type: ignore[assignment]
    gradient: GradientConfig = GradientConfig.model_validate({})
    realization_filters: tuple[RealizationFilterInstance, ...] = ()
    function_estimators: tuple[FunctionEstimatorInstance, ...] = ()
    samplers: tuple[SamplerInstance, ...] = ()
    variable_transforms: tuple[VariableTransformInstance, ...] = ()
    objective_transforms: tuple[ObjectiveTransformInstance, ...] = ()
    nonlinear_constraint_transforms: tuple[
        NonlinearConstraintTransformInstance, ...
    ] = ()
    names: dict[str, tuple[str | int, ...]] = {}

    _locked: bool = PrivateAttr(default=False)

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
    )

    @model_validator(mode="after")
    def _check_linear_constraints(self) -> Self:
        if self.linear_constraints is not None and (
            self.linear_constraints.coefficients.shape[0] > 0
            and self.linear_constraints.coefficients.shape[1]
            != self.variables.variable_count
        ):
            msg = f"the coefficients matrix should have {self.variables.variable_count} columns"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _defaults(self) -> Self:
        updates: dict[str, Any] = {}
        if not self.function_estimators:
            function_estimator_config = FunctionEstimatorConfig.model_validate({})
            updates["function_estimators"] = (
                get_plugin(
                    "function_estimator", method=function_estimator_config.method
                ).create(function_estimator_config),
            )
        if not self.samplers:
            sampler_config = SamplerConfig.model_validate({})
            updates["samplers"] = (
                get_plugin("sampler", method=sampler_config.method).create(
                    sampler_config
                ),
            )
        if updates:
            return self.model_copy(update=updates)
        return self

    @model_validator(mode="after")
    def _initialize_variable_transforms(self) -> Self:
        for idx, item in enumerate(self.variable_transforms):
            item.init(
                np.asarray(self.variables.mask & (self.variables.transforms == idx))
            )

        if self.variable_transforms:
            lower_bounds = self.variables.lower_bounds
            upper_bounds = self.variables.upper_bounds
            magnitudes = self.variables.perturbation_magnitudes
            for transform in self.variable_transforms:
                lower_bounds = transform.to_optimizer(lower_bounds)
                upper_bounds = transform.to_optimizer(upper_bounds)
                magnitudes = transform.magnitudes_to_optimizer(magnitudes)
            absolute = self.variables.perturbation_types == PerturbationType.ABSOLUTE
            updated_variables = self.variables.model_copy(
                update={
                    "lower_bounds": immutable_array(lower_bounds),
                    "upper_bounds": immutable_array(upper_bounds),
                    "perturbation_magnitudes": immutable_array(
                        np.where(
                            absolute,
                            magnitudes,
                            self.variables.perturbation_magnitudes,
                        )
                    ),
                }
            )
            object.__setattr__(self, "variables", updated_variables)  # noqa: PLC2801

            if self.linear_constraints is not None:
                coefficients = self.linear_constraints.coefficients
                lower_bounds = self.linear_constraints.lower_bounds
                upper_bounds = self.linear_constraints.upper_bounds

                for transform in self.variable_transforms:
                    coefficients, lower_bounds, upper_bounds = (
                        transform.linear_constraints_to_optimizer(
                            coefficients, lower_bounds, upper_bounds
                        )
                    )
                updated_linear_constraints = self.model_copy(
                    update={
                        "coefficients": immutable_array(coefficients),
                        "lower_bounds": immutable_array(lower_bounds),
                        "upper_bounds": immutable_array(upper_bounds),
                    }
                )

                object.__setattr__(  # noqa: PLC2801
                    self, "linear_constraints", updated_linear_constraints
                )

        return self

    @model_validator(mode="after")
    def _initialize_objective_transforms(self) -> Self:
        for idx, item in enumerate(self.objective_transforms):
            mask = np.asarray(self.objectives.transforms == idx)
            item.init(mask)
        return self

    @model_validator(mode="after")
    def _initialize_nonlinear_constraint_transforms(self) -> Self:
        if self.nonlinear_constraints is not None:
            for idx, item in enumerate(self.nonlinear_constraint_transforms):
                mask = np.asarray(self.nonlinear_constraints.transforms == idx)
                item.init(mask)
        return self

    @model_validator(mode="wrap")  # type: ignore[arg-type]
    def _pass_context_unchanged(self, handler: Any) -> Any:  # noqa: ANN401
        if isinstance(self, EnOptContext):
            return self
        return handler(self)

    def lock(self) -> None:
        """Lock the object to prevent sharing and re-use.

        Raises:
            RuntimeError: If the object is already locked.
        """
        with _global_lock:
            if self._locked:
                msg = "The EnOptContext object has already been used."
                raise RuntimeError(msg)
            object.__setattr__(self, "_locked", True)  # noqa: PLC2801
