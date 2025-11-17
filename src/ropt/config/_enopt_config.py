"""The `EnOptConfig` configuration class."""

from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, model_validator

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
        names:                 Optional mapping of axis types to names.
    """

    variables: VariablesConfig
    objectives: ObjectiveFunctionsConfig = ObjectiveFunctionsConfig.model_validate({})
    linear_constraints: LinearConstraintsConfig | None = None
    nonlinear_constraints: NonlinearConstraintsConfig | None = None
    realizations: RealizationsConfig = RealizationsConfig.model_validate({})
    optimizer: OptimizerConfig = OptimizerConfig.model_validate({})
    gradient: GradientConfig = GradientConfig.model_validate({})
    realization_filters: tuple[RealizationFilterConfig, ...] = ()
    function_estimators: tuple[FunctionEstimatorConfig, ...] = ()
    samplers: tuple[SamplerConfig, ...] = ()
    names: dict[str, tuple[str | int, ...]] = {}

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
            updates["function_estimators"] = (
                FunctionEstimatorConfig.model_validate({}),
            )
        if not self.samplers:
            updates["samplers"] = (SamplerConfig.model_validate({}),)
        if updates:
            return self.model_copy(update=updates)
        return self

    @model_validator(mode="wrap")  # type: ignore[arg-type]
    def _pass_enopt_config_unchanged(self, handler: Any) -> Any:  # noqa: ANN401
        if isinstance(self, EnOptConfig):
            return self
        return handler(self)
