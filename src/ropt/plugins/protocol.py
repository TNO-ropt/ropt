"""This module defines the protocol to be followed by plugins."""

from __future__ import annotations

from typing import Protocol


class PluginProtocol(Protocol):
    """Protocol for plugin classes.

    `ropt`  plugins should adhere to the `Plugin` protocol, which specifies an
    `is_supported` method.
    """

    def is_supported(self, method: str) -> bool:
        """Check wether a given method is supported.

        Args:
            method: The method name

        Returns:
            True if the method is supported.
        """
