"""Annotated types for Pydantic models providing input conversion and validation."""

from collections.abc import Sequence
from collections.abc import Sequence as AbstractSequence
from collections.abc import Set as AbstractSet
from typing import Annotated, Any, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BeforeValidator, PlainValidator

from ropt.function_estimator import FunctionEstimator
from ropt.plugins.manager import get_plugin
from ropt.realization_filter import RealizationFilter
from ropt.sampler import Sampler
from ropt.transforms import (
    NonlinearConstraintTransform,
    ObjectiveTransform,
    VariableTransform,
)

from ._function_estimator_config import FunctionEstimatorConfig
from ._realization_filter_config import RealizationFilterConfig
from ._sampler_config import SamplerConfig
from ._transform_config import (
    NonlinearConstraintTransformConfig,
    ObjectiveTransformConfig,
    VariableTransformConfig,
)
from ._utils import immutable_array

T = TypeVar("T")


def _convert_1d_array(array: ArrayLike | None) -> NDArray[np.float64] | None:
    if array is None:
        return array
    return immutable_array(array, dtype=np.float64, ndmin=1)


def _convert_1d_array_intc(array: ArrayLike | None) -> NDArray[np.intc] | None:
    if array is None:
        return array
    return immutable_array(array, dtype=np.intc, ndmin=1)


def _convert_1d_array_bool(
    array: ArrayLike | None,
) -> NDArray[np.bool_] | None:
    if array is None:
        return array
    return immutable_array(array, dtype=np.bool_, ndmin=1)


def _convert_2d_array(array: ArrayLike | None) -> NDArray[np.float64] | None:
    if array is None:
        return array
    return immutable_array(array, dtype=np.float64, ndmin=2)


def _convert_enum_array(array: ArrayLike | None) -> NDArray[np.ubyte] | None:
    if array is None:
        return array
    return immutable_array(array, dtype=np.ubyte, ndmin=1)


def _convert_set(value: T | set[T] | Sequence[T]) -> set[T]:
    if isinstance(value, str):
        return {value}
    return set(value) if isinstance(value, AbstractSequence | AbstractSet) else {value}


def _convert_tuple(value: T | Sequence[T]) -> tuple[T, ...]:
    if isinstance(value, str):
        return (value,)
    return tuple(value) if isinstance(value, AbstractSequence) else (value,)


def _convert_sampler(value: Sampler | SamplerConfig | dict[str, Any]) -> Sampler:
    if isinstance(value, Sampler):
        return value
    if isinstance(value, SamplerConfig):
        result = get_plugin("sampler", method=value.method).create(value)
        assert isinstance(result, Sampler)
        return result
    if isinstance(value, dict):
        sampler_config = SamplerConfig.model_validate(value)
        result = get_plugin("sampler", method=sampler_config.method).create(
            sampler_config
        )
        assert isinstance(result, Sampler)
        return result
    msg = "Value must be a Sampler instance, a SamplerConfig instance, or a dict."
    raise ValueError(msg)


def _convert_realization_filter(
    value: RealizationFilter | RealizationFilterConfig | dict[str, Any],
) -> RealizationFilter:
    if isinstance(value, RealizationFilter):
        return value
    if isinstance(value, RealizationFilterConfig):
        result = get_plugin("realization_filter", method=value.method).create(value)
        assert isinstance(result, RealizationFilter)
        return result
    if isinstance(value, dict):
        realization_filter_config = RealizationFilterConfig.model_validate(value)
        result = get_plugin(
            "realization_filter", method=realization_filter_config.method
        ).create(realization_filter_config)
        assert isinstance(result, RealizationFilter)
        return result
    msg = "Value must be a RealizationFilter instance, a RealizationFilterConfig instance, or a dict."
    raise ValueError(msg)


def _convert_function_estimator(
    value: FunctionEstimator | FunctionEstimatorConfig | dict[str, Any],
) -> FunctionEstimator:
    if isinstance(value, FunctionEstimator):
        return value
    if isinstance(value, FunctionEstimatorConfig):
        result = get_plugin("function_estimator", method=value.method).create(value)
        assert isinstance(result, FunctionEstimator)
        return result
    if isinstance(value, dict):
        function_estimator_config = FunctionEstimatorConfig.model_validate(value)
        result = get_plugin(
            "function_estimator", method=function_estimator_config.method
        ).create(function_estimator_config)
        assert isinstance(result, FunctionEstimator)
        return result
    msg = "Value must be a FunctionEstimator instance, a FunctionEstimatorConfig instance, or a dict."
    raise ValueError(msg)


def _convert_variable_transform(
    value: VariableTransform | VariableTransformConfig | dict[str, Any],
) -> VariableTransform:
    if isinstance(value, VariableTransform):
        return value
    if isinstance(value, VariableTransformConfig):
        result = get_plugin("variable_transform", method=value.method).create(value)
        assert isinstance(result, VariableTransform)
        return result
    if isinstance(value, dict):
        variable_transform_config = VariableTransformConfig.model_validate(value)
        result = get_plugin(
            "variable_transform", method=variable_transform_config.method
        ).create(variable_transform_config)
        assert isinstance(result, VariableTransform)
        return result
    msg = "Value must be a VariableTransform instance, a VariableTransformConfig instance, or a dict."
    raise ValueError(msg)


def _convert_objective_transform(
    value: ObjectiveTransform | ObjectiveTransformConfig | dict[str, Any],
) -> ObjectiveTransform:
    if isinstance(value, ObjectiveTransform):
        return value
    if isinstance(value, ObjectiveTransformConfig):
        result = get_plugin("objective_transform", method=value.method).create(value)
        assert isinstance(result, ObjectiveTransform)
        return result
    if isinstance(value, dict):
        objective_transform_config = ObjectiveTransformConfig.model_validate(value)
        result = get_plugin(
            "objective_transform", method=objective_transform_config.method
        ).create(objective_transform_config)
        assert isinstance(result, ObjectiveTransform)
        return result
    msg = "Value must be an ObjectiveTransform instance, an ObjectiveTransformConfig instance, or a dict."
    raise ValueError(msg)


def _convert_nonlinear_constraint_transform(
    value: NonlinearConstraintTransform
    | NonlinearConstraintTransformConfig
    | dict[str, Any],
) -> NonlinearConstraintTransform:
    if isinstance(value, NonlinearConstraintTransform):
        return value
    if isinstance(value, NonlinearConstraintTransformConfig):
        result = get_plugin(
            "nonlinear_constraint_transform", method=value.method
        ).create(value)
        assert isinstance(result, NonlinearConstraintTransform)
        return result
    if isinstance(value, dict):
        nonlinear_constraint_transform_config = (
            NonlinearConstraintTransformConfig.model_validate(value)
        )
        result = get_plugin(
            "nonlinear_constraint_transform",
            method=nonlinear_constraint_transform_config.method,
        ).create(nonlinear_constraint_transform_config)
        assert isinstance(result, NonlinearConstraintTransform)
        return result
    msg = "Value must be a NonlinearConstraintTransform instance, a NonlinearConstraintTransformConfig instance, or a dict."
    raise ValueError(msg)


Array1D = Annotated[NDArray[np.float64], BeforeValidator(_convert_1d_array)]
"""Convert to an immutable 1D numpy array of floating point values."""

Array2D = Annotated[NDArray[np.float64], BeforeValidator(_convert_2d_array)]
"""Convert to an immutable 2D numpy array of floating point values."""

ArrayEnum = Annotated[NDArray[np.ubyte], BeforeValidator(_convert_enum_array)]
"""Convert to an immutable numpy array of numerical enumeration values."""

Array1DInt = Annotated[NDArray[np.intc], BeforeValidator(_convert_1d_array_intc)]
"""Convert to an immutable 1D numpy array of integer values."""

Array1DBool = Annotated[NDArray[np.bool_], BeforeValidator(_convert_1d_array_bool)]
"""Convert to an immutable 1D numpy array of boolean values."""

ItemOrSet = Annotated[set[T], BeforeValidator(_convert_set)]
"""Convert to single value to a set containing that value, passes sets unchanged."""

ItemOrTuple = Annotated[tuple[T, ...], BeforeValidator(_convert_tuple)]
"""Convert to single value to a tuple containing that value, passes sets unchanged."""

SamplerInstance = Annotated[Sampler, PlainValidator(_convert_sampler)]
"""Validate that the value is an instance of a Sampler."""

RealizationFilterInstance = Annotated[
    RealizationFilter, PlainValidator(_convert_realization_filter)
]
"""Validate that the value is an instance of a RealizationFilter."""

FunctionEstimatorInstance = Annotated[
    FunctionEstimator, PlainValidator(_convert_function_estimator)
]
"""Validate that the value is an instance of a FunctionEstimator."""

VariableTransformInstance = Annotated[
    VariableTransform, PlainValidator(_convert_variable_transform)
]
"""Validate that the value is an instance of a VariableTransform."""

ObjectiveTransformInstance = Annotated[
    ObjectiveTransform, PlainValidator(_convert_objective_transform)
]
"""Validate that the value is an instance of an ObjectiveTransform."""

NonlinearConstraintTransformInstance = Annotated[
    NonlinearConstraintTransform,
    PlainValidator(_convert_nonlinear_constraint_transform),
]
"""Validate that the value is an instance of a NonlinearConstraintTransform."""
