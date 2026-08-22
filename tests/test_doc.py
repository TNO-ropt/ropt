from pathlib import Path

import pytest

from ropt.backend.scipy import SCIPY_OPTIONS_SCHEMA, _gen_capability_table
from ropt.config.options import gen_options_table

_SNIPPET_DIR = Path(__file__).parent.parent / "docs" / "snippets"


def _check_snippet(name: str, generated: str) -> None:
    msg = "Regenerate from docs/snippets using: python -m ropt.backend.scipy"
    md_file = _SNIPPET_DIR / name
    if not md_file.exists():
        pytest.fail(f"File not found: {md_file}\n{msg}")
    if md_file.read_text().strip() != generated.strip():
        pytest.fail(f"{name} does not match the generated version.\n{msg}")


def test_scipy_options_table() -> None:
    _check_snippet("scipy.md", gen_options_table(SCIPY_OPTIONS_SCHEMA))


def test_scipy_capability_table() -> None:
    _check_snippet("scipy_capabilities.md", _gen_capability_table())
