"""SciPy optimizer plugin implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ropt.backend.scipy import SUPPORTED_SCIPY_METHODS, SciPyBackend

from ._base import BackendPlugin

if TYPE_CHECKING:
    from ropt.config import BackendConfig


class SciPyBackendPlugin(BackendPlugin):
    """The SciPy backend plugin class."""

    @classmethod
    def create(cls, backend_config: BackendConfig) -> SciPyBackend:
        """Create a SciPyBackend instance.

        Args:
            backend_config: The backend configuration.

        Returns:
            A new `SciPyBackend`.
        """
        return SciPyBackend(backend_config)

    @classmethod
    def is_supported(cls, method: str) -> bool:  # ruff: ignore[undocumented-public-method]
        return method.lower() in (SUPPORTED_SCIPY_METHODS | {"default"})
