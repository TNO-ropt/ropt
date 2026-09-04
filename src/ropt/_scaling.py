"""Arithmetic for scaling and unscaling values.

Values are divided by their scale on the way to the optimizer and multiplied by
it on the way back. Scales are positive, so the map preserves order and sign;
whether an objective is minimized or maximized is a separate setting, applied to
aggregated objectives only.

Variables also carry an offset, so the map is affine rather than a pure
change of units. That is why two kinds of quantity unscale separately:

- A *value* is a quantity in its own right, such as a variable or an
  objective. The offset applies to it.
- A *difference* is the distance between two values, such as a gradient or the
  gap between a variable and its bound. The offset cancels out, so a difference
  is only ever multiplied by the scale.

Objectives and constraints have no offset and pass none.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


def scale(
    values: NDArray[np.float64],
    scales: NDArray[np.float64],
    offsets: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Scale values.

    Args:
        values:  The unscaled values.
        scales:  The scales to apply.
        offsets: The offsets to subtract, if any.

    Returns:
        The scaled values.
    """
    if offsets is None:
        return values / scales
    return (values - offsets) / scales


def unscale_value(
    values: NDArray[np.float64],
    scales: NDArray[np.float64],
    offsets: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Unscale values.

    Args:
        values:  The scaled values.
        scales:  The scales to apply.
        offsets: The offsets to add back, if any.

    Returns:
        The unscaled values.
    """
    if offsets is None:
        return values * scales
    return values * scales + offsets


def unscale_diff(
    diffs: NDArray[np.float64], scales: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Unscale differences.

    Args:
        diffs:  The scaled differences.
        scales: The scales to apply.

    Returns:
        The unscaled differences.
    """
    return diffs * scales
