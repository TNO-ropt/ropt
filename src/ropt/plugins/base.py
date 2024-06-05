"""This module defines the abstract base class for plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Plugin(ABC):
    """Abstract base class for plugins.

    `ropt`  plugins should derive from this base class, which specifies an
    `is_supported` method.
    """

    @abstractmethod
    def is_supported(self, method: str) -> bool:
        """Check wether a given method is supported.

        Args:
            method: The method name

        Returns:
            True if the method is supported.
        """
