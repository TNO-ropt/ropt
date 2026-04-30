from pathlib import Path

import pytest

from ropt.backend.scipy import SCIPY_OPTIONS_SCHEMA
from ropt.config.options import gen_options_table


def test_scipy_options_table() -> None:
    msg = "Regenerate using: python -m ropt.backend.scipy "
    md_file = Path(__file__).parent.parent / "docs" / "snippets" / "scipy.md"
    if not md_file.exists():
        pytest.fail(f"File not found: {md_file}\n{msg}")
    saved = md_file.read_text()
    generated = gen_options_table(SCIPY_OPTIONS_SCHEMA)
    if saved.strip() != generated.strip():
        pytest.fail(
            f"The generated options table does not match the saved version.\n{msg}"
        )
