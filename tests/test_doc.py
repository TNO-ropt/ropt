from pathlib import Path

import pytest

from ropt.config.options import gen_options_table
from ropt.plugins.optimizer.scipy import _OPTIONS_SCHEMA


def test_scipy_options_table() -> None:
    msg = "Regenerate using: python -m ropt.plugins.optimizer.scipy "
    md_file = Path(__file__).parent.parent / "docs" / "snippets" / "scipy.md"
    if not md_file.exists():
        pytest.fail(f"File not found: {md_file}\n{msg}")
    with md_file.open() as fp:
        saved = fp.read()
    generated = gen_options_table(_OPTIONS_SCHEMA)
    if saved.strip() != generated.strip():
        pytest.fail(
            f"The generated options table does not match the saved version.\n{msg}"
        )
