"""Annotated types for Pydantic models providing input conversion and validation.

These types leverage Pydantic's `BeforeValidator` to automatically convert
input values (like lists or scalars) into standardized, immutable NumPy arrays
or Python collections (sets, tuples) during model initialization.

**NumPy Array Types:**

- [`Array1D`][ropt.config.validated_types.Array1D]: Converts input to an
  immutable 1D `np.float64` array.
- [`Array2D`][ropt.config.validated_types.Array2D]: Converts input to an
  immutable 2D `np.float64` array.
- [`ArrayEnum`][ropt.config.validated_types.ArrayEnum]: Converts input to an
  immutable 1D `np.ubyte` array (suitable for integer enum values).
- [`Array1DInt`][ropt.config.validated_types.Array1DInt]: Converts input to an
  immutable 1D `np.intc` array.
- [`Array1DBool`][ropt.config.validated_types.Array1DBool]: Converts input to an
  immutable 1D `np.bool_` array.

**Collection Types:**

- [`ItemOrSet[T]`][ropt.config.validated_types.ItemOrSet]: Ensures the value is a
  `set[T]`, converting single items or sequences.
- [`ItemOrTuple[T]`][ropt.config.validated_types.ItemOrTuple]: Ensures the value is a
  `tuple[T, ...]`, converting single items or sequences.
"""

from typing import Annotated, TypeVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BeforeValidator

from .utils import (
    _convert_1d_array,
    _convert_1d_array_bool,
    _convert_1d_array_intc,
    _convert_2d_array,
    _convert_enum_array,
    _convert_set,
    _convert_tuple,
)

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


T = TypeVar("T")
ItemOrSet = Annotated[set[T], BeforeValidator(_convert_set)]
"""Convert to single value to a set containing that value, passes sets unchanged."""

ItemOrTuple = Annotated[tuple[T, ...], BeforeValidator(_convert_tuple)]
"""Convert to single value to a tuple containing that value, passes sets unchanged."""
