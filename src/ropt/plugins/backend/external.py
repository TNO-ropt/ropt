"""External optimizers plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ropt.backend.external import ExternalBackend
from ropt.plugins.manager import get_plugin_name

from ._base import BackendPlugin

if TYPE_CHECKING:
    from ropt.config import BackendConfig


class ExternalBackendPlugin(BackendPlugin):
    """The external optimizer plugin class."""

    @classmethod
    def create(cls, backend_config: BackendConfig) -> ExternalBackend:
        """Create an ExternalBackend instance.

        Args:
            backend_config: The backend configuration.

        Returns:
            A new `ExternalBackend`.
        """
        return ExternalBackend(backend_config)

    @classmethod
    def is_supported(cls, method: str) -> bool:  # ruff: ignore[undocumented-public-method]
        return get_plugin_name("backend", method) is not None

    @classmethod
    def allows_discovery(cls) -> bool:  # ruff: ignore[undocumented-public-method]
        return False
