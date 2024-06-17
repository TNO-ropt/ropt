"""Utilities for checking configuration values."""

import sys
from collections import Counter
from enum import IntEnum
from typing import Any, Optional, Tuple, Type, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BeforeValidator

if sys.version_info >= (3, 9):
    from typing import Annotated
else:
    from typing_extensions import Annotated


def normalize(array: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize a vector.

        Normalize the sum of the values to one.

    Args:
        array: The input array

    Returns:
        ValueError: The normalized array
    """
    if array.sum() < np.finfo(np.float64).eps:
        msg = "the sum of weights is not positive"
        raise ValueError(msg)
    return immutable_array(array / array.sum())


def immutable_array(
    array_like: ArrayLike,
    **kwargs: Any,  # noqa: ANN401
) -> NDArray[Any]:
    """Make an immutable array.

    Converts the input to an array and makes it immutable.`

    Args:
        array_like: The input
        kwargs    : Additional keyword arguments for array conversion

    Returns:
        The immutable array.
    """
    array = np.array(array_like, **kwargs)
    array.setflags(write=False)
    return array


def broadcast_arrays(*args: Any) -> Tuple[NDArray[Any], ...]:  # noqa: ANN401
    """Broadcast a set of arrays and makes them immutable.

    Args:
        args: The input arrays

    Returns:
        The broadcasted immutable arrays.
    """
    results = np.broadcast_arrays(*args)
    return tuple(immutable_array(result) for result in results)


def broadcast_1d_array(array: NDArray[Any], name: str, size: int) -> NDArray[Any]:
    """Broadcast the input array to an 1D array of given size.

    Args:
        array: The input array
        name:  The name of the array, used in an error message
        size:  The size of the result

    Returns:
        An 1D array of the requested size.
    """
    if size == 0:
        return immutable_array([], dtype=array.dtype)
    try:
        return np.broadcast_to(immutable_array(array), (size,))
    except ValueError as err:
        msg = f"{name} cannot be broadcasted to a length of {size}"
        raise ValueError(msg) from err


def check_enum_values(value: NDArray[np.ubyte], enum_type: Type[IntEnum]) -> None:
    """Check if an enum value is valid.

    Given an array of byte integers, check of the values are within the range of
    values of the given enum.

    Args:
        value:     The enum values
        enum_type: The type to check

    Raises:
        ValueError: If the array contains an invalid value.
    """
    min_enum = min(item.value for item in enum_type)
    max_enum = max(item.value for item in enum_type)
    if np.any(value < min_enum) or np.any(value > max_enum):
        msg = "invalid enumeration value"
        raise ValueError(msg)


def _convert_1d_array(array: Optional[ArrayLike]) -> Optional[NDArray[np.float64]]:
    if array is None:
        return array
    return immutable_array(array, dtype=np.float64, ndmin=1)


def _convert_1d_array_intc(array: Optional[ArrayLike]) -> Optional[NDArray[np.intc]]:
    if array is None:
        return array
    return immutable_array(array, dtype=np.intc, ndmin=1)


def _convert_1d_array_bool(
    array: Optional["ArrayLike"],
) -> Optional[NDArray[np.bool_]]:
    if array is None:
        return array
    return immutable_array(array, dtype=np.bool_, ndmin=1)


def _convert_2d_array(array: Optional[ArrayLike]) -> Optional[NDArray[np.float64]]:
    if array is None:
        return array
    return immutable_array(array, dtype=np.float64, ndmin=2)


def _convert_enum_array(array: Optional[ArrayLike]) -> Optional[NDArray[np.ubyte]]:
    if array is None:
        return array
    return immutable_array(array, dtype=np.ubyte, ndmin=1)


def _convert_indices(array: Optional[ArrayLike]) -> Optional[NDArray[np.intc]]:
    if array is None:
        return array
    return cast(
        NDArray[np.intc],
        np.unique(immutable_array(array, dtype=np.intc, ndmin=1)),
    )


def _check_duplicates(names: Optional[Tuple[Any, ...]]) -> Optional[Tuple[Any, ...]]:
    if names is not None:
        counts = Counter(names)
        duplicates = [name for name, count in counts.items() if count > 1]
        if duplicates:
            raise ValueError("duplicate names: " + ", ".join(duplicates))
    return names


if sys.version_info >= (3, 9):
    Array1D = Annotated[NDArray[np.float64], BeforeValidator(_convert_1d_array)]
    Array2D = Annotated[NDArray[np.float64], BeforeValidator(_convert_2d_array)]
    ArrayIndices = Annotated[NDArray[np.intc], BeforeValidator(_convert_indices)]
    ArrayEnum = Annotated[NDArray[np.ubyte], BeforeValidator(_convert_enum_array)]
    Array1DInt = Annotated[NDArray[np.intc], BeforeValidator(_convert_1d_array_intc)]
    Array1DBool = Annotated[NDArray[np.bool_], BeforeValidator(_convert_1d_array_bool)]
    UniqueNames = Annotated[Tuple[Any, ...], BeforeValidator(_check_duplicates)]
else:
    Array1D = Annotated[ArrayLike, BeforeValidator(_convert_1d_array)]
    Array2D = Annotated[ArrayLike, BeforeValidator(_convert_2d_array)]
    ArrayIndices = Annotated[ArrayLike, BeforeValidator(_convert_indices)]
    ArrayEnum = Annotated[ArrayLike, BeforeValidator(_convert_enum_array)]
    Array1DInt = Annotated[ArrayLike, BeforeValidator(_convert_1d_array_intc)]
    Array1DBool = Annotated[ArrayLike, BeforeValidator(_convert_1d_array_bool)]
    UniqueNames = Annotated[Tuple[Any, ...], BeforeValidator(_check_duplicates)]
