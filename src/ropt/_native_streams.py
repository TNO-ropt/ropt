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


def _standard_streams(libc: ctypes.CDLL) -> list[ctypes.c_void_p]:
    # glibc and musl export `stdout` and `stderr`, macOS `__stdoutp` and `__stderrp`.
    for names in (("stdout", "stderr"), ("__stdoutp", "__stderrp")):
        try:
            return [ctypes.c_void_p.in_dll(libc, name) for name in names]
        except ValueError:
            continue
    return []  # pragma: no cover - other C libraries


_libc = _load_libc()
_streams = [] if _libc is None else _standard_streams(_libc)


def flush_python_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        if stream is not None:
            stream.flush()


def flush_native_streams() -> None:
    # Output written by compiled code sits in libc's buffers, which
    # `sys.stdout.flush()` knows nothing about. Without this it drains after the
    # file descriptors have been switched back, and lands in the wrong place.
    if _libc is None:
        return
    if not _streams:
        # Flushes every stream, but must lock each one to do so, and so waits
        # for any thread that is blocked reading one.
        _libc.fflush(None)
        return
    for stream in _streams:
        _libc.fflush(stream)


def redirect_descriptor(stack: ExitStack, fd: int, target: int) -> None:
    saved = os.dup(fd)
    # Undone in reverse order: `fd` is restored before its copy is closed.
    stack.callback(os.close, saved)
    stack.callback(os.dup2, saved, fd)
    os.dup2(target, fd)
