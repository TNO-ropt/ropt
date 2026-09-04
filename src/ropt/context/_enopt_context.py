"""The `EnOptContext` configuration class."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator

from ropt._scaling import to_optimizer
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
    RealizationFilterInstances,
    SamplerInstances,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

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
    names: dict[str, tuple[str | int, ...]] = {}

    _locked: bool = PrivateAttr(default=False)

    # The scales that are actually applied: the configured `scales`, multiplied
    # by an estimated factor once auto-scaling has run.
    _objective_scales: NDArray[np.float64] = PrivateAttr()
    _constraint_scales: NDArray[np.float64] | None = PrivateAttr()
    _auto_scales_set: bool = PrivateAttr(default=False)

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
    )

    def get_objective_scales(self) -> NDArray[np.float64]:
        """Return the scale applied to each objective.

        Objectives are divided by their scale before they reach the optimizer,
        and multiplied by it again before they are reported. The scales are the
        configured `scales`, multiplied by an estimated factor if auto-scaling
        is enabled and has run.

        Returns:
            The objective scales.
        """
        return self._objective_scales

    def get_constraint_scales(self) -> NDArray[np.float64] | None:
        """Return the scale applied to each nonlinear constraint.

        Returns:
            The constraint scales, or `None` if there are no constraints.
        """
        return self._constraint_scales

    def get_nonlinear_constraint_bounds(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
        """Return the nonlinear constraint bounds in the optimizer domain.

        The bounds are scaled together with the constraint values, so that the
        configured constraint is the constraint that is solved. Scales are
        positive, so the bounds keep their order.

        Returns:
            The lower and upper bounds, or `None` if there are no constraints.
        """
        if self.nonlinear_constraints is None:
            return None
        scales = self._constraint_scales
        assert scales is not None
        return (
            to_optimizer(self.nonlinear_constraints.lower_bounds, scales),
            to_optimizer(self.nonlinear_constraints.upper_bounds, scales),
        )

    def _needs_auto_scales(self) -> bool:
        return not self._auto_scales_set and (
            self.objectives.auto_scale
            or (
                self.nonlinear_constraints is not None
                and bool(self.nonlinear_constraints.auto_scale.any())
            )
        )

    def _set_auto_scales(
        self,
        objectives: NDArray[np.float64] | None,
        constraints: NDArray[np.float64] | None,
    ) -> None:
        # The estimated factors are filled in once and then fixed for the rest
        # of the run, so that every batch is scaled the same way and results
        # stay comparable.
        if self._auto_scales_set:
            msg = "The estimated scales have already been set."
            raise RuntimeError(msg)
        if objectives is not None:
            self._objective_scales = immutable_array(
                self._objective_scales * objectives
            )
        if constraints is not None:
            assert self._constraint_scales is not None
            self._constraint_scales = immutable_array(
                self._constraint_scales * constraints
            )
        self._auto_scales_set = True

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
    def _create_scales(self) -> Self:
        self._objective_scales = immutable_array(self.objectives.scales)
        self._constraint_scales = (
            None
            if self.nonlinear_constraints is None
            else immutable_array(self.nonlinear_constraints.scales)
        )
        return self

    @model_validator(mode="after")
    def _scale_variables_and_constraints(self) -> Self:
        scales = self.variables.scales
        offsets = self.variables.offsets

        # The bounds and the absolute perturbation magnitudes describe the
        # variables, so they move to the optimizer domain with them. A relative
        # magnitude is a fraction of the bound range, which the affine map
        # leaves alone.
        absolute = self.variables.perturbation_types == PerturbationType.ABSOLUTE
        magnitudes = np.where(
            absolute,
            self.variables.perturbation_magnitudes / scales,
            self.variables.perturbation_magnitudes,
        )
        updated_variables = self.variables.model_copy(
            update={
                "lower_bounds": immutable_array(
                    to_optimizer(self.variables.lower_bounds, scales, offsets)
                ),
                "upper_bounds": immutable_array(
                    to_optimizer(self.variables.upper_bounds, scales, offsets)
                ),
                "perturbation_magnitudes": immutable_array(magnitudes),
            }
        )
        object.__setattr__(self, "variables", updated_variables)  # ruff: ignore[unnecessary-dunder-call]

        if self.linear_constraints is not None:
            object.__setattr__(  # ruff: ignore[unnecessary-dunder-call]
                self,
                "linear_constraints",
                _scale_linear_constraints(
                    self.linear_constraints, scales, offsets, self.variables.mask
                ),
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


def _scale_linear_constraints(
    config: LinearConstraintsConfig,
    scales: NDArray[np.float64],
    offsets: NDArray[np.float64],
    mask: NDArray[np.bool_],
) -> LinearConstraintsConfig:
    # Substituting x = scale * y + offset is a change of variables rather than a
    # rescaling: it leaves the distance from a point to each bound unchanged.
    coefficients = config.coefficients * scales
    shift = np.matmul(config.coefficients, offsets)
    lower_bounds = config.lower_bounds - shift
    upper_bounds = config.upper_bounds - shift

    # Scaling the equations is a separate step, and it comes second: the
    # estimate below is only meaningful once the change of variables has been
    # made.
    row_scales = (
        config.scales
        * _estimate_equation_scales(coefficients, lower_bounds, upper_bounds, mask)
        if config.auto_scale
        else config.scales
    )

    return config.model_copy(
        update={
            "coefficients": immutable_array(coefficients / row_scales[:, np.newaxis]),
            "lower_bounds": immutable_array(lower_bounds / row_scales),
            "upper_bounds": immutable_array(upper_bounds / row_scales),
            "scales": immutable_array(row_scales),
        }
    )


def _estimate_equation_scales(
    coefficients: NDArray[np.float64],
    lower_bounds: NDArray[np.float64],
    upper_bounds: NDArray[np.float64],
    mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    # Fixed variables are eliminated before the optimizer sees the problem, so
    # their coefficients must not inflate the estimate.
    largest = np.max(np.abs(coefficients[:, mask]), axis=-1, initial=0.0)
    for bounds in (lower_bounds, upper_bounds):
        largest = np.maximum(
            largest, np.where(np.isfinite(bounds), np.abs(bounds), 0.0)
        )
    # An all-zero equation is one that `get_masked_linear_constraints` drops.
    # Dividing it by its own estimate would turn its coefficients into NaN.
    return np.where(largest > 0.0, largest, 1.0)
