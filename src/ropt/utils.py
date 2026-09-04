"""Optional helpers for scripts and applications that use `ropt`.

Nothing in `ropt` calls anything here. Three kinds of helper live here: queries
about the installed plugins, for code that builds configurations dynamically or
checks them before starting a long run; a converter from variable bounds to the
scales and offsets that a configuration expects; and escape hatches for problems
that come from outside the library, offered because the fix is easy to get wrong
and hard to find.
"""

from __future__ import annotations

import signal
from typing import TYPE_CHECKING, Any

import numpy as np

from ropt.config import BackendConfig
from ropt.plugins.manager import get_plugin, get_plugin_name

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

__all__ = [
    "get_plugin_name",
    "restore_keyboard_interrupt",
    "scales_and_offsets_from_bounds",
    "validate_backend_options",
]


def validate_backend_options(method: str, options: dict[str, Any] | list[str]) -> None:
    """Validate the optimizer-specific options for a given method.

    `method` is either `"plugin-name/method-name"` or just `"method-name"`; see
    [Plugin Discovery](../utilities/plugin_discovery.md) for both forms.

    Args:
        method:  The specific optimization method name.
        options: The dictionary or a list of strings of options.
    """
    plugin = get_plugin("backend", method)
    backend_config = BackendConfig.model_validate(
        {"method": method, "options": options}
    )
    plugin.create(backend_config).validate_options()


def restore_keyboard_interrupt() -> None:
    """Make Ctrl-C interrupt a waiting program again.

    Some third-party extension modules set the process-wide `SA_RESTART` flag
    on `SIGINT` when they are imported, after which Ctrl-C no longer breaks
    into a program that is waiting. This clears the flag, and is a no-op on
    platforms that do not have one.

    Entirely optional: call it at the top of a script, after the imports, only
    if Ctrl-C stops working. See
    [Keyboard Interrupts](../utilities/keyboard_interrupt.md).
    """
    if not hasattr(signal, "siginterrupt"):
        return
    # `siginterrupt` rather than `signal.signal`: it leaves the installed
    # handler in place, so a package that chained its own keeps working.
    interrupt_system_calls = True  # the opposite of SA_RESTART
    signal.siginterrupt(signal.SIGINT, interrupt_system_calls)


def scales_and_offsets_from_bounds(
    lower_bounds: ArrayLike,
    upper_bounds: ArrayLike,
    target_range: tuple[float, float] = (0.0, 1.0),
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Derive variable scales and offsets that map bounds onto a target range.

    The scales and offsets returned here can be passed straight to the `scales`
    and `offsets` fields of a
    [`VariablesConfig`][ropt.config.VariablesConfig] object. They define the map
    $y = (x - o)/s$ that sends `lower_bounds` to the start of `target_range` and
    `upper_bounds` to its end, which puts variables of wildly different
    magnitudes on a common footing for the optimizer.

    The bounds must be finite and the ranges must not be empty, since a variable
    that cannot vary has no scale.

    Args:
        lower_bounds: The lower bounds of the variables.
        upper_bounds: The upper bounds of the variables.
        target_range: The range to map the bounds onto (default: 0 to 1).

    Returns:
        The scales and the offsets.

    Raises:
        ValueError: If the bounds are not finite, or if either range is empty.
    """
    lower = np.asarray(lower_bounds, dtype=np.float64)
    upper = np.asarray(upper_bounds, dtype=np.float64)
    lower, upper = np.broadcast_arrays(lower, upper)

    if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
        msg = "The variable bounds must be finite."
        raise ValueError(msg)
    if np.any(upper <= lower):
        msg = "The variable bounds must define a non-empty range."
        raise ValueError(msg)
    if target_range[1] <= target_range[0]:
        msg = "The target range must be non-empty."
        raise ValueError(msg)

    scales = (upper - lower) / (target_range[1] - target_range[0])
    return scales, lower - target_range[0] * scales
