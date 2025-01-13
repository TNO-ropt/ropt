from __future__ import annotations

from typing import TYPE_CHECKING

from ropt.plugins import PluginManager
from ropt.plugins.optimizer.base import OptimizerCallback, OptimizerPlugin
from ropt.plugins.optimizer.scipy import SciPyOptimizerPlugin

if TYPE_CHECKING:
    from ropt.config.enopt import EnOptConfig


class TestPlugin(OptimizerPlugin):
    def create(self, _0: EnOptConfig, _1: OptimizerCallback) -> None:  # type: ignore[override]
        pass

    def is_supported(self, method: str) -> bool:
        return method.lower() in {"slsqp", "test"}


def test_default_plugins() -> None:
    plugin_manager = PluginManager()

    plugin = plugin_manager.get_plugin("optimizer", "slsqp")
    assert isinstance(plugin, SciPyOptimizerPlugin)


def test_default_plugins_full_spec() -> None:
    plugin_manager = PluginManager()

    plugin = plugin_manager.get_plugin("optimizer", "scipy/slsqp")
    assert isinstance(plugin, SciPyOptimizerPlugin)


def test_added_plugin() -> None:
    plugin_manager = PluginManager()
    plugin_manager.add_plugin("optimizer", "test", TestPlugin())

    plugin = plugin_manager.get_plugin("optimizer", "test")
    assert isinstance(plugin, TestPlugin)

    plugin = plugin_manager.get_plugin("optimizer", "slsqp")
    assert isinstance(plugin, SciPyOptimizerPlugin)

    plugin = plugin_manager.get_plugin("optimizer", "test/slsqp")
    assert isinstance(plugin, TestPlugin)


def test_added_plugin_prioritize() -> None:
    plugin_manager = PluginManager()
    plugin_manager.add_plugin("optimizer", "test", TestPlugin(), prioritize=True)

    plugin = plugin_manager.get_plugin("optimizer", "test")
    assert isinstance(plugin, TestPlugin)

    plugin = plugin_manager.get_plugin("optimizer", "slsqp")
    assert isinstance(plugin, TestPlugin)

    plugin = plugin_manager.get_plugin("optimizer", "scipy/slsqp")
    assert isinstance(plugin, SciPyOptimizerPlugin)
