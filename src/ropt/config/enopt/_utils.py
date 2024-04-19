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
    if array.sum() < np.finfo(np.float64).eps:
        msg = "the sum of weights is not positive"
        raise ValueError(msg)
    return immutable_array(array / array.sum())


def immutable_array(
    array_like: ArrayLike,
    **kwargs: Any,  # noqa: ANN401
) -> NDArray[Any]:
    array = np.array(array_like, **kwargs)
    array.setflags(write=False)
    return array


def broadcast_arrays(*args: Any) -> Tuple[NDArray[Any], ...]:  # noqa: ANN401
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


def check_enum_values(value: NDArray[np.ubyte], enum_type: Type[IntEnum]) -> None:
    min_enum = min(item.value for item in enum_type)
    max_enum = max(item.value for item in enum_type)
    if np.any(value < min_enum) or np.any(value > max_enum):
        msg = "invalid enumeration value"
        raise ValueError(msg)


def _convert_1d_array(array: Optional[ArrayLike]) -> Optional[NDArray[np.float64]]:
    if array is None:
        return array
    return immutable_array(array, dtype=np.float64, ndmin=1, copy=False)


def _convert_1d_array_intc(array: Optional[ArrayLike]) -> Optional[NDArray[np.intc]]:
    if array is None:
        return array
    return immutable_array(array, dtype=np.intc, ndmin=1, copy=False)


def _convert_1d_array_bool(
    array: Optional["ArrayLike"],
) -> Optional[NDArray[np.bool_]]:
    if array is None:
        return array
    return immutable_array(array, dtype=np.bool_, ndmin=1, copy=False)


def _convert_2d_array(array: Optional[ArrayLike]) -> Optional[NDArray[np.float64]]:
    if array is None:
        return array
    return immutable_array(array, dtype=np.float64, ndmin=2, copy=False)


def _convert_enum_array(array: Optional[ArrayLike]) -> Optional[NDArray[np.ubyte]]:
    if array is None:
        return array
    return immutable_array(array, dtype=np.ubyte, ndmin=1, copy=False)


def _convert_indices(array: Optional[ArrayLike]) -> Optional[NDArray[np.intc]]:
    if array is None:
        return array
    return cast(
        NDArray[np.intc],
        np.unique(immutable_array(array, dtype=np.intc, ndmin=1, copy=False)),
    )


def _check_duplicates(names: Optional[Tuple[Any, ...]]) -> Optional[Tuple[Any, ...]]:
    if names is not None:
        counts = Counter(names)
        duplicates = [name for name, count in counts.items() if count > 1]
        if duplicates:
            raise ValueError("duplicate names: " + ", ".join(duplicates))
    return names


Array1D = Annotated[
    NDArray[np.float64],
    BeforeValidator(_convert_1d_array),
]
Array2D = Annotated[
    NDArray[np.float64],
    BeforeValidator(_convert_2d_array),
]
ArrayIndices = Annotated[
    NDArray[np.intc],
    BeforeValidator(_convert_indices),
]
ArrayEnum = Annotated[
    NDArray[np.ubyte],
    BeforeValidator(_convert_enum_array),
]
Array1DInt = Annotated[
    NDArray[np.intc],
    BeforeValidator(_convert_1d_array_intc),
]
Array1DBool = Annotated[
    NDArray[np.bool_],
    BeforeValidator(_convert_1d_array_bool),
]
UniqueNames = Annotated[
    Tuple[Any, ...],
    BeforeValidator(_check_duplicates),
]
