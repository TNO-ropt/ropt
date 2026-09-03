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
from ropt.exceptions import WorkflowError
from ropt.plugins.manager import get_plugin

from ._validated_types import (  # ruff: ignore[typing-only-first-party-import]
    BackendInstance,
    FunctionEstimatorInstances,
    NonlinearConstraintTransformInstance,
    ObjectiveTransformInstance,
    RealizationFilterInstances,
    SamplerInstances,
    VariableTransformInstance,
)

# Module-global rather than per-instance: a lock stored on the model would make
# the context unpicklable, and the external backend pickles it to the child.
_global_lock = threading.Lock()


class EnOptContext(BaseModel):
    """The primary context object for a single optimization run.

    `EnOptContext` holds all information needed to run an ensemble-based
    optimization: variables, objectives, constraints, realizations, gradient
    settings, samplers, filters, and the optimizer/backend. It is constructed
    from plain Python dicts or config objects and validated on creation.

    See the [Configuration guide](../optimizer_setup/configuration.md)
    for an in-depth description of broadcasting rules, index-based sharing of
    plugin instances, the `names` attribute, and how dicts are resolved into
    plugin instances.

    Warning:
        `EnOptContext` objects are immutable after construction. Do not attempt
        to serialize and round-trip them (for example to and from JSON): `numpy` arrays
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
        realization_filters:             Realization filter plugin instances, by key.
                                         A sequence is keyed by position.
        function_estimators:             Function estimator plugin instances, by key.
                                         A sequence is keyed by position.
        samplers:                        Sampler plugin instances, by key. A sequence
                                         is keyed by position.
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
    realization_filters: RealizationFilterInstances = {}
    function_estimators: FunctionEstimatorInstances = {}
    samplers: SamplerInstances = {}
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
    def _check_nonlinear_constraint_transforms(self) -> Self:
        # `variables` and `objectives` always exist, so only this chain can be orphaned.
        if self.nonlinear_constraint_transforms and self.nonlinear_constraints is None:
            msg = "nonlinear constraint transforms need nonlinear constraints"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _defaults(self) -> Self:
        updates: dict[str, Any] = {}
        if not self.function_estimators:
            function_estimator_config = FunctionEstimatorConfig.model_validate({})
            updates["function_estimators"] = {
                "0": get_plugin(
                    "function_estimator", method=function_estimator_config.method
                ).create(function_estimator_config)
            }
        if not self.samplers:
            sampler_config = SamplerConfig.model_validate({})
            updates["samplers"] = {
                "0": get_plugin("sampler", method=sampler_config.method).create(
                    sampler_config
                )
            }
        if updates:
            return self.model_copy(update=updates)
        return self

    @model_validator(mode="after")
    def _check_references(self) -> Self:
        sections = [
            ("variables", self.variables, ("samplers",)),
            (
                "objectives",
                self.objectives,
                ("realization_filters", "function_estimators"),
            ),
        ]
        if self.nonlinear_constraints is not None:
            sections.append(
                (
                    "nonlinear_constraints",
                    self.nonlinear_constraints,
                    ("realization_filters", "function_estimators"),
                )
            )
        for section, config, fields in sections:
            for field in fields:
                defined = getattr(self, field)
                for key in getattr(config, field):
                    if key is not None and key not in defined:
                        known = (
                            "defined keys are "
                            + ", ".join(repr(item) for item in defined)
                            if defined
                            else "no keys are defined"
                        )
                        msg = f"{section}.{field}: unknown key {key!r}; {known}"
                        raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _initialize_variable_transforms(self) -> Self:
        for item in self.variable_transforms:
            item.set_free_mask(self.variables.mask)

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
            object.__setattr__(self, "variables", updated_variables)  # ruff: ignore[unnecessary-dunder-call]

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
                updated_linear_constraints = self.linear_constraints.model_copy(
                    update={
                        "coefficients": immutable_array(coefficients),
                        "lower_bounds": immutable_array(lower_bounds),
                        "upper_bounds": immutable_array(upper_bounds),
                    }
                )

                object.__setattr__(  # ruff: ignore[unnecessary-dunder-call]
                    self, "linear_constraints", updated_linear_constraints
                )

        return self

    @model_validator(mode="wrap")  # type: ignore[arg-type]
    def _pass_context_unchanged(self, handler: Any) -> Any:  # ruff: ignore[any-type]
        if isinstance(self, EnOptContext):
            return self
        return handler(self)

    def lock(self) -> None:
        """Lock the object to prevent sharing and re-use.

        Raises:
            WorkflowError: If the object is already locked.
        """
        with _global_lock:
            if self._locked:
                msg = "The EnOptContext object has already been used."
                raise WorkflowError(msg)
            object.__setattr__(self, "_locked", True)  # ruff: ignore[unnecessary-dunder-call]
