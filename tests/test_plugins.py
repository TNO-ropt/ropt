from __future__ import annotations

from typing import TYPE_CHECKING

from ropt.plugins import PluginManager
from ropt.plugins.optimizer.base import OptimizerCallback, OptimizerPlugin
from ropt.plugins.optimizer.scipy import SciPyOptimizerPlugin

if TYPE_CHECKING:
    from ropt.config.enopt import EnOptConfig


class MockedPlugin(OptimizerPlugin):
    @classmethod
    def create(cls, _0: EnOptConfig, _1: OptimizerCallback) -> None:  # type: ignore[override]
        pass

    @classmethod
    def is_supported(cls, method: str) -> bool:
        return method.lower() in {"slsqp", "test"}


def test_default_plugins() -> None:
    plugin_manager = PluginManager()

    plugin = plugin_manager.get_plugin("optimizer", "slsqp")
    assert issubclass(plugin, SciPyOptimizerPlugin)


def test_default_plugins_full_spec() -> None:
    plugin_manager = PluginManager()

    plugin = plugin_manager.get_plugin("optimizer", "scipy/slsqp")
    assert issubclass(plugin, SciPyOptimizerPlugin)


def test_added_plugin() -> None:
    plugin_manager = PluginManager()
    plugin_manager.add_plugin("optimizer", "test", MockedPlugin)

    plugin = plugin_manager.get_plugin("optimizer", "test")
    assert issubclass(plugin, MockedPlugin)

    plugin = plugin_manager.get_plugin("optimizer", "slsqp")
    assert issubclass(plugin, SciPyOptimizerPlugin)

    plugin = plugin_manager.get_plugin("optimizer", "test/slsqp")
    assert issubclass(plugin, MockedPlugin)


def test_added_plugin_prioritize() -> None:
    plugin_manager = PluginManager()
    plugin_manager.add_plugin("optimizer", "test", MockedPlugin, prioritize=True)

    plugin = plugin_manager.get_plugin("optimizer", "test")
    assert issubclass(plugin, MockedPlugin)

    plugin = plugin_manager.get_plugin("optimizer", "slsqp")
    assert issubclass(plugin, MockedPlugin)

    plugin = plugin_manager.get_plugin("optimizer", "scipy/slsqp")
    assert issubclass(plugin, SciPyOptimizerPlugin)
