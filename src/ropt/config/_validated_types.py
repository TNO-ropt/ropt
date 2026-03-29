"""Annotated types for Pydantic models providing input conversion and validation."""

from collections.abc import Sequence
from collections.abc import Sequence as AbstractSequence
from collections.abc import Set as AbstractSet
from typing import Annotated, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BeforeValidator

from ropt._utils import immutable_array

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
