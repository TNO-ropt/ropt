"""Utilities for handing work to an event loop from another thread.

These, with the worker-id sets in `ThreadingExecutor` and `EventDispatcher`, are
how this package answers "where am I running?" — deliberately, rather than with
`threading.local` or `contextvars`.

Ambient state must never decide **which work runs, where, or under what
configuration**. The context variables behind the old `ropt.simple` blocks
existed only to shorten call sites, and cost a separate nested-run entry point,
handler stealing and a one-executor rule until they were removed.

Two things are not that, and are why these helpers exist. Reading the caller's
own location to *refuse* a call that could only hang leaves the operation's
meaning fixed by its arguments: it can turn a deadlock into an error, and
nothing else. And state that merely carries diagnostics alongside an operation
its arguments already determine changes nothing either. Both should say so where
they are introduced.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


def schedule(
    loop: asyncio.AbstractEventLoop | None,
    callback: Callable[..., object],
    *args: object,
) -> bool:
    if loop is None:
        return False
    try:
        loop.call_soon_threadsafe(callback, *args)
    except RuntimeError:
        return False
    return True


def on_loop_thread(loop: asyncio.AbstractEventLoop | None) -> bool:
    try:
        return asyncio.get_running_loop() is loop
    except RuntimeError:
        return False
