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
    def create(cls, backend_config: BackendConfig) -> SciPyBackend:  # noqa: D102
        return SciPyBackend(backend_config)

    @classmethod
    def is_supported(cls, method: str) -> bool:  # noqa: D102
        return method.lower() in (SUPPORTED_SCIPY_METHODS | {"default"})
