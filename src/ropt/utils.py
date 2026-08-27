"""Optional helpers for scripts and applications that use `ropt`.

Nothing in `ropt` calls anything here. These are escape hatches for problems
that come from outside the library, offered because the fix is easy to get
wrong and hard to find.
"""

from __future__ import annotations

import signal

__all__ = ["restore_keyboard_interrupt"]


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
