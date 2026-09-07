"""Flushing of output buffers that live below the Python level."""

from __future__ import annotations

import ctypes


def _load_libc() -> ctypes.CDLL | None:
    try:
        return ctypes.CDLL(None)
    except (OSError, TypeError):  # pragma: no cover - non-POSIX platforms
        return None


_libc = _load_libc()


def flush_native_streams() -> None:
    # Output written by compiled code sits in libc's buffers, which
    # `sys.stdout.flush()` knows nothing about. Without this it drains after the
    # file descriptors have been switched back, and lands in the wrong place.
    if _libc is not None:
        _libc.fflush(None)
