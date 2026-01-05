# ruff: noqa: SLF001
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import pytest
from pydantic import ValidationError

from ropt.config.options import OptionsSchemaModel
from ropt.plugins import PluginManager, plugin_manager
from ropt.plugins.optimizer.base import OptimizerPlugin
from ropt.plugins.optimizer.scipy import SciPyOptimizerPlugin

if TYPE_CHECKING:
    from ropt.config import EnOptConfig
    from ropt.optimization import OptimizerCallback


class MockedPlugin1(OptimizerPlugin):
    @classmethod
    def create(cls, _0: EnOptConfig, _1: OptimizerCallback) -> None:  # type: ignore[override]
        pass

    @classmethod
    def is_supported(cls, method: str) -> bool:
        return method.lower() == "test"


class MockedPlugin2(OptimizerPlugin):
    @classmethod
    def create(cls, _0: EnOptConfig, _1: OptimizerCallback) -> None:  # type: ignore[override]
        pass

    @classmethod
    def is_supported(cls, method: str) -> bool:
        return method.lower() == "test"


class MockedPluginWithValidation(MockedPlugin1):
    @classmethod
    def validate_options(
        cls, method: str, options: dict[str, Any] | list[str] | None
    ) -> None:
        OptionsSchemaModel.model_validate(
            {
                "methods": {
                    "Test": {
                        "options": {
                            "a": float | str,
                            "b": Literal["foo", "bar"],
                        },
                        "url": "https://example.org",
                    },
                },
            }
        ).get_options_model(method).model_validate(options)


def test_default_plugins() -> None:
    plugin = plugin_manager.get_plugin("optimizer", "slsqp")
    assert issubclass(plugin, SciPyOptimizerPlugin)


def test_default_plugins_full_spec() -> None:
    plugin = plugin_manager.get_plugin("optimizer", "scipy/slsqp")
    assert issubclass(plugin, SciPyOptimizerPlugin)


def test_added_ambiguous_method(monkeypatch: Any) -> None:
    manager = PluginManager()
    monkeypatch.setattr(manager, "_init", lambda: None)
    manager._add_plugin("optimizer", "test1", MockedPlugin1)
    manager._add_plugin("optimizer", "test2", MockedPlugin2)

    with pytest.raises(
        ValueError, match="Ambiguous method: 'test' is available in multiple plugins"
    ):
        manager.get_plugin("optimizer", "test")


def test_validate_options(monkeypatch: Any) -> None:
    manager = PluginManager()
    monkeypatch.setattr(manager, "_init", lambda: None)
    manager._add_plugin("optimizer", "test", MockedPluginWithValidation)
    plugin = manager.get_plugin("optimizer", "test")
    assert issubclass(plugin, MockedPluginWithValidation)
    plugin.validate_options("test", {"a": 1.0})
    plugin.validate_options("test", {"a": "foo"})
    with pytest.raises(ValidationError, match="Input should be a valid number"):
        plugin.validate_options("test", {"a": []})
    plugin.validate_options("Test", {"b": "foo"})
    with pytest.raises(ValidationError, match="Input should be 'foo' or 'bar'"):
        plugin.validate_options("TEST", {"b": "wrong"})
    with pytest.raises(
        ValidationError, match=r"Unknown or unsupported option\(s\): `c`, `d`"
    ):
        plugin.validate_options("test", {"c": 1, "d": "foo"})
