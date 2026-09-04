"""Utilities for checking and converting configuration and result values."""

from enum import IntEnum
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def normalize(array: NDArray[np.float64]) -> NDArray[np.float64]:
    # A negative weight would flip the sign of what it weighs, which is a
    # direction, not a weight. Zero stays legal: it disables an entry.
    if np.any(array < 0.0):
        msg = "Weights must not be negative"
        raise ValueError(msg)
    if array.sum() < np.finfo(np.float64).eps:
        msg = "The sum of weights is not positive"
        raise ValueError(msg)
    return immutable_array(array / array.sum())


def zero_failures(values: NDArray[np.float64]) -> NDArray[np.float64]:
    # Only NaN marks a failure; an infinity is a value and must survive.
    return np.nan_to_num(values, posinf=np.inf, neginf=-np.inf)


def apply_direction(
    values: NDArray[np.float64], maximize: NDArray[np.bool_]
) -> NDArray[np.float64]:
    # Minimizing the negation of a value maximizes the value. Applied per
    # objective, so that one sign never has to stand in for several.
    if not maximize.any():
        return values
    return np.where(maximize, -values, values)


def immutable_array(
    array_like: ArrayLike,
    **kwargs: Any,  # ruff: ignore[any-type]
) -> NDArray[Any]:
    array = np.array(array_like, **kwargs)
    array.setflags(write=False)
    return array


def broadcast_arrays(*args: Any) -> tuple[NDArray[Any], ...]:  # ruff: ignore[any-type]
    results = np.broadcast_arrays(*args)
    return tuple(immutable_array(result) for result in results)


def broadcast_1d_array(array: NDArray[Any], name: str, size: int) -> NDArray[Any]:
    if size == 0:
        return immutable_array([], dtype=array.dtype)
    try:
        return np.broadcast_to(immutable_array(array), (size,))
    except ValueError as err:
        msg = f"{name} cannot be broadcasted to a length of {size}"
        raise ValueError(msg) from err


def broadcast_keys(
    keys: tuple[str | None, ...], name: str, size: int
) -> tuple[str | None, ...]:
    if size == 0:
        return ()
    if len(keys) == 1:
        return keys * size
    if len(keys) != size:
        msg = f"{name} cannot be broadcasted to a length of {size}"
        raise ValueError(msg)
    return keys


def check_scales(scales: NDArray[np.float64], name: str, size: int) -> NDArray[Any]:
    # A scale is a change of units, so it is positive. Direction is a separate
    # setting: folding it into the scale would make one number mean two things.
    if np.any(scales <= 0.0):
        msg = f"{name} must be positive"
        raise ValueError(msg)
    return broadcast_1d_array(scales, name, size)


def check_enum_values(value: NDArray[np.ubyte], enum_type: type[IntEnum]) -> None:
    min_enum = min(item.value for item in enum_type)
    max_enum = max(item.value for item in enum_type)
    if np.any(value < min_enum) or np.any(value > max_enum):
        msg = "invalid enumeration value"
        raise ValueError(msg)
