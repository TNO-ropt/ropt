# ruff: file-ignore[private-member-access]
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

import pytest
from pydantic import ValidationError

from ropt.backend import Backend
from ropt.backend.scipy import SciPyBackend
from ropt.config.options import OptionsSchemaModel
from ropt.plugins.manager import (
    PluginManager,
    get_plugin,
    get_plugin_name,
    register_plugin,
)

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ropt.config import BackendConfig
    from ropt.context import EnOptContext
    from ropt.core import OptimizerCallback
    from ropt.plugins import MethodSpec


class MockedPlugin1(Backend):
    methods: ClassVar[MethodSpec] = {"test"}

    def __init__(self, _0: BackendConfig) -> None:
        pass

    def init(self, _0: EnOptContext, _1: OptimizerCallback) -> None:
        pass

    def start(self, _0: NDArray[np.float64]) -> None:
        pass

    def validate_options(self) -> None:
        pass


class MockedPlugin2(MockedPlugin1):
    pass


class MockedPluginWithoutMethods(Backend):
    def __init__(self, _0: BackendConfig) -> None:
        pass

    def init(self, _0: EnOptContext, _1: OptimizerCallback) -> None:
        pass

    def start(self, _0: NDArray[np.float64]) -> None:
        pass

    def validate_options(self) -> None:
        pass


class MockedPluginWithValidation(MockedPlugin1):
    @classmethod
    def validate_options(  # type: ignore[override]
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
    assert issubclass(get_plugin("backend", "slsqp"), SciPyBackend)


def test_default_plugins_full_spec() -> None:
    assert issubclass(get_plugin("backend", "scipy/slsqp"), SciPyBackend)


@pytest.mark.parametrize(
    "method", ["scipy/slsqp", "SciPy/slsqp", "SCIPY/slsqp", "SCIPY/SLSQP"]
)
def test_the_plugin_name_is_case_insensitive(method: str) -> None:
    assert get_plugin_name("backend", method) == "scipy"
    assert issubclass(get_plugin("backend", method), SciPyBackend)


def test_added_ambiguous_method(monkeypatch: Any) -> None:
    manager = PluginManager()
    monkeypatch.setattr(manager, "_init", lambda: None)
    manager._add_plugin("backend", "test1", MockedPlugin1)
    manager._add_plugin("backend", "test2", MockedPlugin2)

    with pytest.raises(ValueError, match="Method 'test' is ambiguous across plugins"):
        manager.get_plugin("backend", "test")


def test_validate_options(monkeypatch: Any) -> None:
    manager = PluginManager()
    monkeypatch.setattr(manager, "_init", lambda: None)
    manager._add_plugin("backend", "test", MockedPluginWithValidation)
    plugin = manager.get_plugin("backend", "test")
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


def test_a_method_set_is_matched_case_insensitively(monkeypatch: Any) -> None:
    manager = PluginManager()
    monkeypatch.setattr(manager, "_init", lambda: None)
    manager._add_plugin("backend", "mocked", MockedPlugin1)
    assert manager.get_plugin_name("backend", "mocked/test") == "mocked"
    assert manager.get_plugin_name("backend", "mocked/TEST") == "mocked"
    assert manager.get_plugin_name("backend", "mocked/other") is None


def test_a_method_reaches_a_predicate_verbatim(monkeypatch: Any) -> None:
    seen: list[str] = []

    def predicate(method: str) -> bool:
        seen.append(method)
        return method == "CamelCase"

    class PredicatePlugin(MockedPlugin1):
        methods: ClassVar[MethodSpec] = staticmethod(predicate)

    manager = PluginManager()
    monkeypatch.setattr(manager, "_init", lambda: None)
    manager._add_plugin("backend", "mocked", PredicatePlugin)

    assert manager.get_plugin_name("backend", "mocked/CamelCase") == "mocked"
    assert manager.get_plugin_name("backend", "mocked/camelcase") is None
    assert seen == ["CamelCase", "camelcase"]


def test_an_undiscoverable_plugin_is_not_found_by_method_alone() -> None:
    assert get_plugin_name("backend", "external/slsqp") == "external"
    assert get_plugin_name("backend", "slsqp") == "scipy"


def test_a_registered_plugin_is_found_like_an_installed_one(
    monkeypatch: Any,
) -> None:
    # Registering targets the module-level manager, so keep it out of the
    # manager the rest of the session shares.
    monkeypatch.setattr("ropt.plugins.manager._plugin_manager", None)

    register_plugin("backend", "Mocked", MockedPlugin1)

    assert get_plugin_name("backend", "mocked/test") == "mocked"
    assert get_plugin_name("backend", "MOCKED/TEST") == "mocked"
    assert get_plugin_name("backend", "test") == "mocked"
    assert get_plugin("backend", "mocked/test") is MockedPlugin1
    assert get_plugin_name("backend", "slsqp") == "scipy"


def test_registering_a_name_again_replaces_it() -> None:
    manager = PluginManager()
    manager.register_plugin("backend", "mocked", MockedPlugin1)
    manager.register_plugin("backend", "mocked", MockedPlugin2)
    assert manager.get_plugin("backend", "mocked/test") is MockedPlugin2


def test_an_installed_plugin_cannot_be_registered_over() -> None:
    manager = PluginManager()
    with pytest.raises(ValueError, match="named `scipy`, it cannot be replaced"):
        manager.register_plugin("backend", "SciPy", MockedPlugin1)
    assert manager.get_plugin("backend", "scipy/slsqp") is SciPyBackend


def test_registering_a_plugin_without_methods_is_rejected() -> None:
    manager = PluginManager()
    with pytest.raises(TypeError, match="`mocked` does not declare the methods"):
        manager.register_plugin("backend", "mocked", MockedPluginWithoutMethods)
    assert manager.get_plugin_name("backend", "mocked/test") is None


def test_registering_a_plugin_of_the_wrong_type_is_rejected() -> None:
    manager = PluginManager()
    with pytest.raises(TypeError, match="Wrong type for sampler plugin `mocked`"):
        manager.register_plugin("sampler", "mocked", MockedPlugin1)
