"""The `EnOptConfig` configuration class."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator

from ropt.enums import PerturbationType
from ropt.plugins.manager import get_plugin

from ._backend_config import BackendConfig
from ._function_estimator_config import FunctionEstimatorConfig
from ._gradient_config import GradientConfig
from ._linear_constraints_config import LinearConstraintsConfig  # noqa: TC001
from ._nonlinear_constraints_config import NonlinearConstraintsConfig  # noqa: TC001
from ._objective_functions_config import ObjectiveFunctionsConfig
from ._optimizer_config import OptimizerConfig
from ._realizations_config import RealizationsConfig
from ._sampler_config import SamplerConfig
from ._utils import immutable_array
from ._validated_types import (  # noqa: TC001
    FunctionEstimatorInstance,
    NonlinearConstraintTransformInstance,
    ObjectiveTransformInstance,
    RealizationFilterInstance,
    SamplerInstance,
    VariableTransformInstance,
)
from ._variables_config import VariablesConfig  # noqa: TC001

if TYPE_CHECKING:
    from collections.abc import Generator


class EnOptConfig(BaseModel):
    """The primary configuration class for an optimization run.

    `EnOptConfig` orchestrates the configuration of an entire optimization
    workflow. It contains nested configuration classes that define specific
    aspects of the optimization, such as variables, objectives, constraints,
    realizations, and the optimizer itself.

    `realization_filters`, `function_estimators`, and `samplers` are configured
    as tuples. Other configuration fields reference these objects by their index
    within the tuples. This makes it possible to share these objects between
    entities. For example, [`VariablesConfig`][ropt.config.VariablesConfig] has
    a `samplers` field, which is an array of indices specifying the sampler to
    use for each variable. If only a single sampler is needed, the `samplers`
    field in `EnOptConfig` should contain a single sampler configuration, and
    the `samplers` field in the `VariablesConfig` configuration contains only
    zeros to specify that each variable should use this entry. In case of
    multiple samplers, multiple sampler configurations are defined, and each
    entry in `samplers` array in `VariablesConfig` points to desired sampler.

    The optional `names` attribute is a dictionary that stores the names of the
    various entities, such as variables, objectives, and constraints. The
    supported name types are defined in the [`AxisName`][ropt.enums.AxisName]
    enumeration. This information is optional, as it is not strictly necessary
    for the optimization, but it can be useful for labeling and interpreting
    results. For instance, when present, it is used to create a multi-index
    results that are exported as data frames.

    Info:
        Many nested configuration classes use `numpy` arrays. These arrays
        typically have a size determined by a configured property (e.g., the
        number of variables) or a size of one. In the latter case, the single
        value is broadcasted to all relevant elements. For example,
        [`VariablesConfig`][ropt.config.VariablesConfig] defines properties like
        initial values and bounds as `numpy` arrays, which must either match the
        number of variables or have a size of one.

    Warning:
        `EnOptConfig` objects are immutable and hold the in-memory configuration
        for an optimization run. For persistence, do not serialize the object
        itself. Instead, store the original dictionary used for its creation.

        Round-trip serialization (e.g., to/from JSON) is not a supported use
        case and may lead to data loss due to the complex types it contains,
        such as `numpy` arrays, or to unexpected behavior because of the
        transformations that are applied to the input upon creation.

    Attributes:
        variables:                       Configuration for the optimization variables.
        objectives:                      Configuration for the objective functions.
        linear_constraints:              Configuration for linear constraints.
        nonlinear_constraints:           Configuration for non-linear constraints.
        realizations:                    Configuration for the realizations.
        optimizer:                       Configuration for the ensemble optimizer.
        backend:                         Configuration for the optimization backend.
        gradient:                        Configuration for gradient calculations.
        realization_filters:             Configuration for realization filters.
        function_estimators:             Configuration for function estimators.
        samplers:                        Configuration for samplers.
        variable_transforms:             Configuration for variable transforms.
        objective_transforms:            Configuration for objective transforms.
        nonlinear_constraint_transforms: Configuration for nonlinear constraint transforms.
        names:                           Optional mapping of axis types to names.
    """

    variables: VariablesConfig
    objectives: ObjectiveFunctionsConfig = ObjectiveFunctionsConfig.model_validate({})
    linear_constraints: LinearConstraintsConfig | None = None
    nonlinear_constraints: NonlinearConstraintsConfig | None = None
    realizations: RealizationsConfig = RealizationsConfig.model_validate({})
    optimizer: OptimizerConfig = OptimizerConfig.model_validate({})
    backend: BackendConfig = BackendConfig.model_validate({})
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

    _lock: threading.Lock | None = PrivateAttr(default=None)

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        frozen=True,
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
    def _pass_enopt_config_unchanged(self, handler: Any) -> Any:  # noqa: ANN401
        if isinstance(self, EnOptConfig):
            return self
        return handler(self)

    @contextmanager
    def lock(self) -> Generator[Self, None, None]:
        """Lock the object to prevent sharing.

        Yields:
            Self: The locked object.

        Raises:
            RuntimeError: If the object is already locked.
        """
        if self._lock is None:
            object.__setattr__(self, "_lock", threading.Lock())  # noqa: PLC2801
        if not self._lock.acquire(blocking=False):
            msg = "The EnOptConfig object is already in use."
            raise RuntimeError(msg)
        try:
            yield self
        finally:
            self._lock.release()
