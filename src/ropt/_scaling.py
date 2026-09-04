"""Arithmetic for mapping values between the user and optimizer domains.

Values are divided by their scale on the way to the optimizer and multiplied by
it on the way back. Scales are positive, so the map preserves order and sign;
whether an objective is minimized or maximized is a separate setting, applied to
aggregated objectives only.

Two kinds of quantity map back, and they are kept apart deliberately: they
coincide today, but would diverge if an offset were ever added, and collapsing
them into a single multiplication is what would make that change expensive.

- A *value* is a quantity in its own right, such as an objective. An offset
  would apply to it.
- A *difference* is the distance between two values, such as a gradient or the
  gap between a constraint and its bound. An offset cancels out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


def to_optimizer(
    values: NDArray[np.float64], scales: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Map values from the user domain to the optimizer domain.

    Args:
        values: The values in the user domain.
        scales: The scales to apply.

    Returns:
        The values in the optimizer domain.
    """
    return values / scales


def value_from_optimizer(
    values: NDArray[np.float64], scales: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Map values from the optimizer domain back to the user domain.

    Args:
        values: The values in the optimizer domain.
        scales: The scales to apply.

    Returns:
        The values in the user domain.
    """
    return values * scales


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
