"""Tests that every module imports on an installation without optional extras."""

from __future__ import annotations

import importlib
import pkgutil
from importlib.util import find_spec
from typing import Final

import pytest

import ropt

_OPTIONAL_PACKAGES: Final = ("pandas", "polars", "cloudpickle", "pysqa")

pytestmark = pytest.mark.skipif(
    any(find_spec(name) is not None for name in _OPTIONAL_PACKAGES),
    reason="only meaningful when no optional package is installed",
)

_UNGUARDED_IMPORTS: Final = frozenset({"ropt.results._pandas", "ropt.results._polars"})


def _module_names() -> list[str]:
    return [
        info.name
        for info in pkgutil.walk_packages(ropt.__path__, "ropt.")
        if info.name not in _UNGUARDED_IMPORTS
    ]


@pytest.mark.parametrize("name", _module_names())
def test_module_imports_without_the_optional_packages(name: str) -> None:
    importlib.import_module(name)
