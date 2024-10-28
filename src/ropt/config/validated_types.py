"""Annotated types for use with Pydantic models.

These types can be used to convert input values to a desired type and guarantee
certain properties. They include types that convert inputs to immutable NumPy
arrays of specified dimension and type:

- [`Array1D`][ropt.config.validated_types.Array1D]: For converting sequences to
  immutable one-dimensional floating-point arrays.
- [`Array2D`][ropt.config.validated_types.Array2D]: For converting sequences to
  immutable two-dimensional floating-point arrays.
- [`ArrayIndices`][ropt.config.validated_types.ArrayIndices]: For converting
  sequences to immutable arrays containing indices of any dimension.
- [`ArrayEnum`][ropt.config.validated_types.ArrayEnum]: For converting
  sequences to values of numerical enumerations of any dimension.
- [`Array1DInt`][ropt.config.validated_types.Array1DInt]: For converting
  sequences to immutable one-dimensional integer arrays.
- [`Array1DBool`][ropt.config.validated_types.Array1DBool]: For converting
  sequences to immutable one-dimensional boolean arrays.

Additionally, the following convenience types create sets or tuples, ensuring
that single values are embedded in a set or tuple, respectively:

- [`ItemOrSet[T]`][ropt.config.validated_types.ItemOrSet]: Create a set of type `T`.
- [`ItemOrTuple[T]`][ropt.config.validated_types.ItemOrTuple]: Create a tuple of type `T`.
"""

import sys
from typing import Any, Set, Tuple, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import AfterValidator, BeforeValidator

if sys.version_info >= (3, 9):
    from typing import Annotated
else:
    from typing_extensions import Annotated

from .utils import (
    _check_duplicates,
    _convert_1d_array,
    _convert_1d_array_bool,
    _convert_1d_array_intc,
    _convert_2d_array,
    _convert_enum_array,
    _convert_indices,
    _convert_set,
    _convert_tuple,
)

if sys.version_info >= (3, 9):
    Array1D = Annotated[NDArray[np.float64], BeforeValidator(_convert_1d_array)]
    """Convert to an immutable 1D numpy array of floating point values."""

    Array2D = Annotated[NDArray[np.float64], BeforeValidator(_convert_2d_array)]
    """Convert to an immutable 2D numpy array of floating point values."""

    ArrayIndices = Annotated[NDArray[np.intc], BeforeValidator(_convert_indices)]
    """Convert to an immutable numpy array of indices."""

    ArrayEnum = Annotated[NDArray[np.ubyte], BeforeValidator(_convert_enum_array)]
    """Convert to an immutable numpy array of numerical enumeration values."""

    Array1DInt = Annotated[NDArray[np.intc], BeforeValidator(_convert_1d_array_intc)]
    """Convert to an immutable 1D numpy array of integer values."""

    Array1DBool = Annotated[NDArray[np.bool_], BeforeValidator(_convert_1d_array_bool)]
    """Convert to an immutable 1D numpy array of boolean values."""

    UniqueNames = Annotated[Tuple[Any, ...], AfterValidator(_check_duplicates)]
    """Check for duplicates in a tuple, raising ValueError if duplicates are found."""

else:
    Array1D = Annotated[ArrayLike, BeforeValidator(_convert_1d_array)]
    Array2D = Annotated[ArrayLike, BeforeValidator(_convert_2d_array)]
    ArrayIndices = Annotated[ArrayLike, BeforeValidator(_convert_indices)]
    ArrayEnum = Annotated[ArrayLike, BeforeValidator(_convert_enum_array)]
    Array1DInt = Annotated[ArrayLike, BeforeValidator(_convert_1d_array_intc)]
    Array1DBool = Annotated[ArrayLike, BeforeValidator(_convert_1d_array_bool)]
    UniqueNames = Annotated[Tuple[Any, ...], AfterValidator(_check_duplicates)]

T = TypeVar("T")
ItemOrSet = Annotated[Set[T], BeforeValidator(_convert_set)]
"""Convert to single value to a set containing that value, passes sets unchanged."""

ItemOrTuple = Annotated[Tuple[T, ...], BeforeValidator(_convert_tuple)]
"""Convert to single value to a tuple containing that value, passes sets unchanged."""
