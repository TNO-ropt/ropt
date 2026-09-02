"""Optional helpers for scripts and applications that use `ropt`.

Nothing in `ropt` calls anything here. Two kinds of helper live here: queries
about the installed plugins, for code that builds configurations dynamically or
checks them before starting a long run; and escape hatches for problems that
come from outside the library, offered because the fix is easy to get wrong and
hard to find.
"""

from __future__ import annotations

import signal
from typing import Any

from ropt.config import BackendConfig
from ropt.plugins.manager import get_plugin, get_plugin_name

__all__ = [
    "get_plugin_name",
    "restore_keyboard_interrupt",
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
