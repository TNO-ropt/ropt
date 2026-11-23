"""Flushing and redirecting of the standard output streams."""

from __future__ import annotations

import ctypes
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from contextlib import ExitStack


def _load_libc() -> ctypes.CDLL | None:
    try:
        return ctypes.CDLL(None)
    except (OSError, TypeError):  # pragma: no cover - non-POSIX platforms
        return None


_libc = _load_libc()


def flush_python_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        if stream is not None:
            stream.flush()


def flush_native_streams() -> None:
    # Output written by compiled code sits in libc's buffers, which
    # `sys.stdout.flush()` knows nothing about. Without this it drains after the
    # file descriptors have been switched back, and lands in the wrong place.
    if _libc is not None:
        _libc.fflush(None)


def redirect_descriptor(stack: ExitStack, fd: int, target: int) -> None:
    saved = os.dup(fd)
    # Undone in reverse order: `fd` is restored before its copy is closed.
    stack.callback(os.close, saved)
    stack.callback(os.dup2, saved, fd)
    os.dup2(target, fd)
