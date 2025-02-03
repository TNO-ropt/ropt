"""The `EnOptConfig` configuration class."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Self

from pydantic import ConfigDict, ValidationInfo, model_validator

from ropt.config.utils import ImmutableBaseModel
from ropt.transforms import OptModelTransforms

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


@dataclass
class EnOptContext:
    """A context object to pass to EnOptConfig.model_validate."""

    transforms: OptModelTransforms = field(default_factory=OptModelTransforms)


class EnOptConfig(ImmutableBaseModel):
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

    The `realization_filters`, `function_estimators`, and `samplers` fields are
    defined as tuples of configurations for realization filter, function
    estimator, and sampler objects, respectively. Other configuration fields
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
        objectives:            Configuration of the objective functions.
        linear_constraints:    Configuration of linear constraints.
        nonlinear_constraints: Configuration of non-linear constraints.
        realizations:          Configuration of the realizations.
        optimizer:             Configuration of the optimizer.
        gradient:              Configuration for gradient calculations.
        realization_filters:   Configuration of realization filters.
        function_estimators:   Configuration of function estimators.
        samplers:              Configuration of samplers.
        original_inputs:       The original input to the constructor.
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
    original_inputs: dict[str, Any] | None = None

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
    )

    @model_validator(mode="after")
    def _linear_constraints(self, info: ValidationInfo) -> Self:
        if self.linear_constraints is not None:
            self._mutable()
            self.linear_constraints = self.linear_constraints.apply_transformation(
                self.variables, info.context
            )
            self._immutable()
        return self

    @model_validator(mode="after")
    def _gradient(self, info: ValidationInfo) -> Self:
        self._mutable()
        self.gradient = self.gradient.fix_perturbations(self.variables, info.context)
        self._immutable()
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
