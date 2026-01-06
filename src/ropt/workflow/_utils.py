"""The plugin manager."""

from __future__ import annotations

from typing import Any

from ropt.plugins import plugin_manager


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
    return plugin_manager.get_plugin_name("sampler", method)


def find_optimizer_plugin(method: str) -> str | None:
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
    return plugin_manager.get_plugin_name("optimizer", method)


def validate_optimizer_options(
    method: str, options: dict[str, Any] | list[str]
) -> None:
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
    plugin_manager.get_plugin("optimizer", method).validate_options(method, options)
