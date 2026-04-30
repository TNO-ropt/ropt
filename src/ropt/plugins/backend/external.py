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
    def create(cls, backend_config: BackendConfig) -> ExternalBackend:  # noqa: D102
        return ExternalBackend(backend_config)

    @classmethod
    def is_supported(cls, method: str) -> bool:  # noqa: D102
        return get_plugin_name("backend", method) is not None

    @classmethod
    def allows_discovery(cls) -> bool:  # noqa: D102
        return False
