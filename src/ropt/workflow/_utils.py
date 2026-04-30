"""The plugin manager."""

from __future__ import annotations

from typing import Any

from ropt.config import BackendConfig
from ropt.plugins.manager import get_plugin, get_plugin_name


def find_sampler_plugin(method: str) -> str | None:
    """Find a sampler plugin for a given method.

    The `method` argument can be specified in two ways:

    1.  **Explicit Plugin:** `"plugin-name/method-name"` checks if the specific
        plugin named `plugin-name` supports `method-name`.
    2.  **Implicit Plugin:** `"method-name"` searches through all discoverable
        plugins to see if any support `method-name`.

    Args:
        method: The method name.

    Returns:
        The name of the plugin that implements the sampler method or `None`.
    """
    return get_plugin_name("sampler", method)


def find_backend_plugin(method: str) -> str | None:
    """Find an optimizer plugin for a given method.

    The `method` argument can be specified in two ways:

    1.  **Explicit Plugin:** `"plugin-name/method-name"` checks if the specific
        plugin named `plugin-name` supports `method-name`.
    2.  **Implicit Plugin:** `"method-name"` searches through all discoverable
        plugins to see if any support `method-name`.

    Args:
        method: The method name.

    Returns:
        The name of the plugin that implements the optimizer method or `None`.
    """
    return get_plugin_name("backend", method)


def validate_backend_options(method: str, options: dict[str, Any] | list[str]) -> None:
    """Validate the optimizer-specific options for a given method.

    The `method` argument can be specified in two ways:

    1.  **Explicit Plugin:** `"plugin-name/method-name"` checks if the specific
        plugin named `plugin-name` supports `method-name`.
    2.  **Implicit Plugin:** `"method-name"` searches through all discoverable
        plugins to see if any support `method-name`.

    Args:
        method:  The specific optimization method name.
        options: The dictionary or a list of strings of options.
    """
    plugin = get_plugin("backend", method)
    backend_config = BackendConfig.model_validate(
        {"method": method, "options": options}
    )
    plugin.create(backend_config).validate_options()
