"""This module provides the default plugin for servers.

**Supported Components:**

  - `function_server`: Server that forwards calculations to an evaluation function.
    ([`DefaultFunctionEvaluatorServer`][ropt.plugins.server._function_server.DefaultFunctionEvaluatorServer])
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from ._function_server import DefaultFunctionEvaluatorServer
from .base import ServerPlugin

if TYPE_CHECKING:
    from .base import Server


_SERVER_OBJECTS: Final[dict[str, type[Server[Any]]]] = {
    "function_server": DefaultFunctionEvaluatorServer,
}


class DefaultServerPlugin(ServerPlugin):
    """The default plugin for creating evaluators.

    This plugin acts as a factory for the standard server implementations
    provided by `ropt`.

    **Supported Servers:**

    - `function_server`: Server that forwards calculations to an evaluation function.
        ([`DefaultFunctionEvaluatorServer`][ropt.plugins.server._function_server.DefaultFunctionEvaluatorServer])
    """

    @classmethod
    def create(
        cls,
        name: str,
        **kwargs: Any,  # noqa: ANN401
    ) -> Server[Any]:
        """Create a server.

        # noqa
        """  # noqa: DOC201, DOC501
        _, _, name = name.lower().rpartition("/")
        server_obj = _SERVER_OBJECTS.get(name)
        if server_obj is not None:
            return server_obj(**kwargs)

        msg = f"Unknown evaluator type: {name}"
        raise TypeError(msg)

    @classmethod
    def is_supported(cls, method: str) -> bool:
        """Check if a method is supported.

        See the [ropt.plugins.base.Plugin][] abstract base class.

        # noqa
        """  # noqa: DOC201
        return method.lower() in _SERVER_OBJECTS
