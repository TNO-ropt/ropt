"""Utilities for checking and converting configuration values.

This module provides helper functions primarily designed for use within Pydantic
model validation logic. These functions facilitate the conversion of
configuration inputs into standardized, immutable NumPy arrays and handle common
validation tasks like checking enum values or broadcasting arrays to required
dimensions.
"""

from collections.abc import Sequence as AbstractSequence
from collections.abc import Set as AbstractSet
from enum import IntEnum
from typing import Any, Sequence, Type, TypeVar, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel


def normalize(array: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize a NumPy array so its elements sum to one.

    Args:
        array: The input NumPy array (1D).

    Returns:
        A new immutable NumPy array with the same shape as the input, where
        the elements have been scaled to sum to 1.0.

    Raises:
        ValueError: If the sum of the input array elements is not positive
                    (i.e., less than or equal to machine epsilon).
    """
    if array.sum() < np.finfo(np.float64).eps:
        msg = "the sum of weights is not positive"
        raise ValueError(msg)
    return immutable_array(array / array.sum())


def immutable_array(
    array_like: ArrayLike,
    **kwargs: Any,  # noqa: ANN401
) -> NDArray[Any]:
    """Convert input to an immutable NumPy array.

    This function takes various array-like inputs (e.g., lists, tuples, other
    NumPy arrays) and converts them into a NumPy array. It then sets the
    `writeable` flag of the resulting array to `False`, making it immutable.

    Args:
        array_like: The input data to convert (e.g., list, tuple, NumPy array).
        kwargs:     Additional keyword arguments passed directly to `numpy.array`.

    Returns:
        A new NumPy array, with its `writeable` flag set to `False`.
    """
    array = np.array(array_like, **kwargs)
    array.setflags(write=False)
    return array


def broadcast_arrays(*args: Any) -> tuple[NDArray[Any], ...]:  # noqa: ANN401
    """Broadcast arrays to a common shape and make them immutable.

    This function takes multiple NumPy arrays (or array-like objects) and uses
    `numpy.broadcast_arrays` to make them conform to a common shape according
    to NumPy's broadcasting rules. Each resulting array is then made immutable
    by setting its `writeable` flag to `False`.

    Args:
        args: A variable number of NumPy arrays or array-like objects.

    Returns:
        A tuple containing the broadcasted, immutable NumPy arrays.
    """
    results = np.broadcast_arrays(*args)
    return tuple(immutable_array(result) for result in results)


def broadcast_1d_array(array: NDArray[Any], name: str, size: int) -> NDArray[Any]:
    """Broadcast an array to a 1D array of a specific size and make it immutable.

    This function takes an input array and attempts to broadcast it to a
    one-dimensional array of the specified `size` using NumPy's broadcasting
    rules. If successful, the resulting array is made immutable.

    This is useful for ensuring configuration parameters (like weights or
    magnitudes) have the correct dimension corresponding to the number of
    variables, objectives, etc., allowing users to provide a single scalar value
    that applies to all elements.

    Args:
        array: The input NumPy array or array-like object.
        name:  A descriptive name for the array (used in error messages).
        size:  The target size (number of elements) for the 1D array.

    Returns:
        A new, immutable 1D NumPy array of the specified `size`.

    Raises:
        ValueError: If the input `array` cannot be broadcast to the target `size`.
    """
    if size == 0:
        return immutable_array([], dtype=array.dtype)
    try:
        return np.broadcast_to(immutable_array(array), (size,))
    except ValueError as err:
        msg = f"{name} cannot be broadcasted to a length of {size}"
        raise ValueError(msg) from err


def check_enum_values(value: NDArray[np.ubyte], enum_type: Type[IntEnum]) -> None:
    """Check if enum values in a NumPy array are valid members of an IntEnum.

    This function verifies that all integer values within the input NumPy array
    correspond to valid members of the specified `IntEnum` type.

    Args:
        value:     A NumPy array containing integer values (typically `np.ubyte`)
                   representing potential enum members.
        enum_type: The `IntEnum` class to validate against.

    Raises:
        ValueError: If any value in the `value` array does not correspond to a
                    member of the `enum_type`.
    """
    min_enum = min(item.value for item in enum_type)
    max_enum = max(item.value for item in enum_type)
    if np.any(value < min_enum) or np.any(value > max_enum):
        msg = "invalid enumeration value"
        raise ValueError(msg)


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


T = TypeVar("T")


def _convert_set(value: T | set[T] | Sequence[T]) -> set[T]:
    if isinstance(value, str):
        return {cast("T", value)}
    return set(value) if isinstance(value, AbstractSequence | AbstractSet) else {value}


def _convert_tuple(value: T | Sequence[T]) -> tuple[T, ...]:
    if isinstance(value, str):
        return (cast("T", value),)
    return tuple(value) if isinstance(value, AbstractSequence) else (value,)


class ImmutableBaseModel(BaseModel):
    """Base model providing manual immutability control.

    This class offers an alternative to Pydantic's `frozen=True` configuration.
    It allows instances to be mutable during initialization (e.g., within
    `@model_validator(mode='after')`) and then explicitly made immutable
    afterwards by calling the `_immutable()` method.

    Immutability is enforced by overriding `__setattr__` to check an internal
    `_is_immutable` flag before allowing attribute modification. The
    `_mutable()` method can be used to temporarily disable immutability if
    needed, though this should be used with caution.

    This approach is particularly useful when complex validation or modification
    logic needs to run *after* the initial Pydantic model initialization, which
    can be problematic with standard frozen models.
    """

    _is_immutable: bool

    def _immutable(self) -> None:
        self._is_immutable = True

    def _mutable(self) -> None:
        self._is_immutable = False

    def __setattr__(self, name: str, value: Any) -> None:  # noqa: ANN401
        """Set an attribute's value, enforcing immutability.

        This method overrides the default attribute setting behavior. If the
        instance has been marked as immutable (via `_immutable()`), it prevents
        any further attribute modifications (except for the internal `_is_immutable`
        flag itself) and raises an `AttributeError`.

        Args:
            name:  The name of the attribute to set.
            value: The value to assign to the attribute.

        Raises:
            AttributeError: If attempting to set an attribute on an immutable instance.
        """
        if name != "_is_immutable" and self._is_immutable:
            msg = f"{self.__class__.__name__} is immutable"
            raise AttributeError(msg)
        super().__setattr__(name, value)
