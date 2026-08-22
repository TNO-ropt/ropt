"""Plugin discovery and validation helpers."""

from __future__ import annotations

from typing import Any

from ropt.config import BackendConfig
from ropt.plugins.manager import get_plugin, get_plugin_name


def find_sampler_plugin(method: str) -> str | None:
    """Find a sampler plugin for a given method.

    `method` is either `"plugin-name/method-name"` or just `"method-name"`; see
    [Plugin Discovery](../utilities/plugin_discovery.md) for both forms.

    Args:
        method: The method name.

    Returns:
        The name of the plugin that implements the sampler method or `None`.
    """
    return get_plugin_name("sampler", method)


def find_backend_plugin(method: str) -> str | None:
    """Find an optimizer plugin for a given method.

    `method` is either `"plugin-name/method-name"` or just `"method-name"`; see
    [Plugin Discovery](../utilities/plugin_discovery.md) for both forms.

    Args:
        method: The method name.

    Returns:
        The name of the plugin that implements the optimizer method or `None`.
    """
    return get_plugin_name("backend", method)


def validate_backend_options(method: str, options: dict[str, Any] | list[str]) -> None:
    """Validate the optimizer-specific options for a given method.

    `method` is either `"plugin-name/method-name"` or just `"method-name"`; see
    [Plugin Discovery](../utilities/plugin_discovery.md) for both forms.

    Args:
        method:  The specific optimization method name.
        options: The dictionary or a list of strings of options.
    """
    plugin = get_plugin("backend", method)
    backend_config = BackendConfig.model_validate(
        {"method": method, "options": options}
    )
    plugin.create(backend_config).validate_options()
