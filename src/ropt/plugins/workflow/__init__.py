"""Plugin functionality for adding workflow objects.

This package contains the abstract base class for workflow plugins, and the
default workflow objects that are part of `ropt`.
"""

from ._callback import DefaultCallbackContext, DefaultCallbackWith
from ._config import DefaultConfigContext

__all__ = [
    "DefaultCallbackContext",
    "DefaultCallbackWith",
    "DefaultConfigContext",
]
