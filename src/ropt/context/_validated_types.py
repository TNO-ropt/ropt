"""Annotated types for Pydantic models providing input conversion and validation."""

from collections.abc import Callable, Sequence
from typing import Annotated, Any, Protocol, Self

from pydantic import BeforeValidator, PlainValidator

from ropt.backend import Backend
from ropt.config import (
    BackendConfig,
    FunctionEstimatorConfig,
    RealizationFilterConfig,
    SamplerConfig,
)
from ropt.function_estimator import FunctionEstimator
from ropt.plugins import PluginType, get_plugin
from ropt.realization_filter import RealizationFilter
from ropt.sampler import Sampler


class _PluginConfig(Protocol):
    method: str

    @classmethod
    def model_validate(cls, obj: Any, /) -> Self:  # ruff: ignore[any-type]
        ...


def _make_validator[ConfigT: _PluginConfig, InstanceT](
    plugin_type: PluginType,
    config_type: type[ConfigT],
    instance_type: type[InstanceT],
    extra: Callable[[InstanceT], None] | None = None,
) -> Callable[[Any], InstanceT]:
    article = "an" if instance_type.__name__[0] in "AEIOU" else "a"

    def _convert(value: Any) -> InstanceT:  # ruff: ignore[any-type]
        if isinstance(value, instance_type):
            result = value
        elif isinstance(value, (config_type, dict)):
            config = (
                value
                if isinstance(value, config_type)
                else config_type.model_validate(value)
            )
            result = get_plugin(plugin_type, method=config.method)(config)
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


def _convert_to_tuple(value: Any) -> Any:  # ruff: ignore[any-type]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(value)
    return (value,)


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
    tuple[SamplerInstance, ...], BeforeValidator(_convert_to_tuple)
]
"""Validate one sampler or a sequence of samplers."""

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
    tuple[RealizationFilterInstance, ...], BeforeValidator(_convert_to_tuple)
]
"""Validate one realization filter or a sequence of realization filters."""

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
    tuple[FunctionEstimatorInstance, ...], BeforeValidator(_convert_to_tuple)
]
"""Validate one function estimator or a sequence of function estimators."""
