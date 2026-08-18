"""Utilities for handing work to an event loop from another thread."""

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
