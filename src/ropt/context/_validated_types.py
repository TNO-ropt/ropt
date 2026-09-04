"""Annotated types for Pydantic models providing input conversion and validation."""

from collections.abc import Callable, Mapping, Sequence
from typing import Annotated, Any, Protocol, Self, TypeVar

from pydantic import BeforeValidator, PlainValidator

from ropt.backend import Backend
from ropt.config import (
    BackendConfig,
    FunctionEstimatorConfig,
    RealizationFilterConfig,
    SamplerConfig,
    VariableTransformConfig,
)
from ropt.function_estimator import FunctionEstimator
from ropt.plugins.manager import PluginType, get_plugin
from ropt.realization_filter import RealizationFilter
from ropt.sampler import Sampler
from ropt.transforms import VariableTransform


class _PluginConfig(Protocol):
    method: str

    @classmethod
    def model_validate(cls, obj: Any, /) -> Self:  # ruff: ignore[any-type]
        ...


_ConfigT = TypeVar("_ConfigT", bound=_PluginConfig)
_InstanceT = TypeVar("_InstanceT")


def _make_validator(
    plugin_type: PluginType,
    config_type: type[_ConfigT],
    instance_type: type[_InstanceT],
    extra: Callable[[_InstanceT], None] | None = None,
) -> Callable[[Any], _InstanceT]:
    article = "an" if instance_type.__name__[0] in "AEIOU" else "a"

    def _convert(value: Any) -> _InstanceT:  # ruff: ignore[any-type]
        if isinstance(value, instance_type):
            result = value
        elif isinstance(value, (config_type, dict)):
            config = (
                value
                if isinstance(value, config_type)
                else config_type.model_validate(value)
            )
            result = get_plugin(plugin_type, method=config.method).create(config)
            assert isinstance(result, instance_type)
        else:
            msg = (
                f"Value must be {article} {instance_type.__name__}, "
                f"{config_type.__name__}, or dict."
            )
            raise ValueError(msg)  # ruff: ignore[type-check-without-type-error]
        if extra is not None:
            extra(result)
        return result

    return _convert


def _validate_backend_options(backend: Backend) -> None:
    backend.validate_options()


def _convert_to_mapping(value: Any) -> Any:  # ruff: ignore[any-type]
    # A sequence is keyed by position, so the integers that used to index it
    # keep selecting the same entry.
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return {str(index): item for index, item in enumerate(value)}
    return value


BackendInstance = Annotated[
    Backend,
    PlainValidator(
        _make_validator("backend", BackendConfig, Backend, _validate_backend_options)
    ),
]
"""Validate that the value is an instance of a Backend."""

SamplerInstance = Annotated[
    Sampler, PlainValidator(_make_validator("sampler", SamplerConfig, Sampler))
]
"""Validate that the value is an instance of a Sampler."""

SamplerInstances = Annotated[
    dict[str, SamplerInstance], BeforeValidator(_convert_to_mapping)
]
"""Validate a mapping of keys to Sampler instances; a sequence is keyed by position."""

RealizationFilterInstance = Annotated[
    RealizationFilter,
    PlainValidator(
        _make_validator(
            "realization_filter", RealizationFilterConfig, RealizationFilter
        )
    ),
]
"""Validate that the value is an instance of a RealizationFilter."""

RealizationFilterInstances = Annotated[
    dict[str, RealizationFilterInstance], BeforeValidator(_convert_to_mapping)
]
"""Validate a mapping of keys to RealizationFilters; a sequence is keyed by position."""

FunctionEstimatorInstance = Annotated[
    FunctionEstimator,
    PlainValidator(
        _make_validator(
            "function_estimator", FunctionEstimatorConfig, FunctionEstimator
        )
    ),
]
"""Validate that the value is an instance of a FunctionEstimator."""

FunctionEstimatorInstances = Annotated[
    dict[str, FunctionEstimatorInstance], BeforeValidator(_convert_to_mapping)
]
"""Validate a mapping of keys to FunctionEstimators; a sequence is keyed by position."""

VariableTransformInstance = Annotated[
    VariableTransform,
    PlainValidator(
        _make_validator(
            "variable_transform", VariableTransformConfig, VariableTransform
        )
    ),
]
"""Validate that the value is an instance of a VariableTransform."""
