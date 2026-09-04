"""Arithmetic for mapping values between the user and optimizer domains.

Values are divided by their scale on the way to the optimizer and multiplied by
it on the way back. Scales are positive, so the map preserves order and sign;
whether an objective is minimized or maximized is a separate setting, applied to
aggregated objectives only.

Variables also carry an offset, so the map is affine rather than a pure
change of units. That is why two kinds of quantity map back separately:

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


def to_optimizer(
    values: NDArray[np.float64],
    scales: NDArray[np.float64],
    offsets: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Map values from the user domain to the optimizer domain.

    Args:
        values:  The values in the user domain.
        scales:  The scales to apply.
        offsets: The offsets to subtract, if any.

    Returns:
        The values in the optimizer domain.
    """
    if offsets is None:
        return values / scales
    return (values - offsets) / scales


def value_from_optimizer(
    values: NDArray[np.float64],
    scales: NDArray[np.float64],
    offsets: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Map values from the optimizer domain back to the user domain.

    Args:
        values:  The values in the optimizer domain.
        scales:  The scales to apply.
        offsets: The offsets to add back, if any.

    Returns:
        The values in the user domain.
    """
    if offsets is None:
        return values * scales
    return values * scales + offsets


def diff_from_optimizer(
    diffs: NDArray[np.float64], scales: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Map differences from the optimizer domain back to the user domain.

    Args:
        diffs:  The differences in the optimizer domain.
        scales: The scales to apply.

    Returns:
        The differences in the user domain.
    """
    return diffs * scales
