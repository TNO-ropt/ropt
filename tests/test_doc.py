import re
from pathlib import Path

import pytest

from ropt.backend.scipy import SCIPY_OPTIONS_SCHEMA, _gen_capability_table
from ropt.config.options import gen_options_table

_ROOT = Path(__file__).parent.parent
_SNIPPET_DIR = _ROOT / "docs" / "snippets"
_EXAMPLES_INDEX = _ROOT / "docs" / "examples" / "index.md"

# Misplaced backticks render as a code span, so no reference reaches
# mkdocs-autorefs and `mkdocs build --strict` stays silent.
_MALFORMED_REF = re.compile(r"\[`[^`\n]*\]\[[^`\n]*`\]")


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


def test_no_malformed_cross_references() -> None:
    found = [
        f"{path.relative_to(_ROOT)}:{line_nr}: {match.group()}"
        for path in [*(_ROOT / "docs").rglob("*.md"), *(_ROOT / "src").rglob("*.py")]
        for line_nr, line in enumerate(path.read_text().splitlines(), start=1)
        for match in _MALFORMED_REF.finditer(line)
    ]
    if found:
        pytest.fail(
            "Malformed cross-references, expected [`X`][path]:\n" + "\n".join(found)
        )


def test_every_example_script_is_listed_in_the_examples_index() -> None:
    index = _EXAMPLES_INDEX.read_text()
    missing = [
        str(path.relative_to(_ROOT))
        for sub_dir in ("simple", "advanced")
        for path in sorted((_ROOT / "examples" / sub_dir).glob("*.py"))
        if str(path.relative_to(_ROOT)) not in index
    ]
    if missing:
        pytest.fail(
            f"Not listed in {_EXAMPLES_INDEX.relative_to(_ROOT)}, so unreachable "
            "from the documentation:\n" + "\n".join(missing)
        )
