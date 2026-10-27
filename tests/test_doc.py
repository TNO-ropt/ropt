import dataclasses
import re
import runpy
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import ropt.results
from ropt import DataFrameHandler, optimize
from ropt.backend.scipy import SCIPY_OPTIONS_SCHEMA, _gen_capability_table
from ropt.config.options import gen_options_table
from ropt.enums import ExitCode
from ropt.results import FunctionResults, GradientResults

_ROOT = Path(__file__).parent.parent
_SNIPPET_DIR = _ROOT / "docs" / "snippets"
_QUICKSTART = _ROOT / "docs" / "getting_started" / "quickstart.md"
_EXIT_CODES = _ROOT / "docs" / "results" / "exit_codes.md"
_EXAMPLES_PAGE = _ROOT / "docs" / "getting_started" / "examples.md"
_HANDLERS = _ROOT / "docs" / "results" / "handlers.md"

# Misplaced backticks render as a code span, so no reference reaches
# mkdocs-autorefs and `mkdocs build --strict` stays silent.
_MALFORMED_REF = re.compile(r"\[`[^`\n]*\]\[[^`\n]*`\]")
_PYTHON_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)
_EXAMPLE_LINK = re.compile(r"examples/\w+\.py")
_TABLE_CODE = re.compile(r"^\| `(\w+)`", re.MULTILINE)
_TABLE_FIELD = re.compile(r"^[ \t]*\| `([\w.<>]+)`", re.MULTILINE)
_COMMENTED_FIELD = re.compile(r'^\s*# "([\w.<>]+)":', re.MULTILINE)

# Attributes of a result that do not export to a table column.
_NOT_TABLE_FIELDS = frozenset({"names", "source_key"})
_TABLE_TYPES = {
    "functions": (FunctionResults, {"batch_id", "realization"}),
    "gradients": (GradientResults, {"batch_id", "realization", "perturbation"}),
}


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


def test_quickstart_program_reaches_the_optimum_it_claims(tmp_path: Path) -> None:
    block = _PYTHON_BLOCK.search(_QUICKSTART.read_text())
    assert block is not None, "the Quickstart no longer contains a python block"
    script = tmp_path / "quickstart.py"
    script.write_text(block.group(1))
    result = runpy.run_path(str(script), run_name="__main__")["result"]
    assert result.results is not None
    assert np.allclose(result.results.variables, 1.0, atol=1e-2)


def test_examples_page_list_every_script() -> None:
    page = _EXAMPLES_PAGE.read_text()
    missing = [
        str(path.relative_to(_ROOT))
        for path in sorted((_ROOT / "examples").glob("*.py"))
        if str(path.relative_to(_ROOT)) not in page
    ]
    if missing:
        pytest.fail(
            "Not listed on the examples page, so unreachable from the "
            "documentation:\n" + "\n".join(missing)
        )


def test_examples_page_have_no_dead_entries() -> None:
    # The scripts are linked by their GitHub URL, which mkdocs never resolves.
    missing = sorted(
        {
            name
            for name in _EXAMPLE_LINK.findall(_EXAMPLES_PAGE.read_text())
            if not (_ROOT / name).exists()
        }
    )
    if missing:
        pytest.fail("Listed but absent from the repository:\n" + "\n".join(missing))


def test_exit_codes_page_tabulates_every_exit_code() -> None:
    # Checked against the table rather than the page, because most names also
    # appear in the prose, which would hide a row that was never added.
    tabulated = set(_TABLE_CODE.findall(_EXIT_CODES.read_text()))
    missing = [code.name for code in ExitCode if code.name not in tabulated]
    if missing:
        pytest.fail(
            "Missing from the table on the exit codes page, so a run can report "
            "a code the manual does not explain:\n" + "\n".join(missing)
        )


def _result_fields(cls: type, prefix: str = "") -> set[str]:
    # Annotations are strings, so a nested result field is found by class name.
    paths: set[str] = set()
    for field in dataclasses.fields(cls):
        path = prefix + field.name
        type_name = str(field.type)
        candidates = [
            getattr(ropt.results, name, None) for name in re.findall(r"\w+", type_name)
        ]
        nested = [
            item
            for item in candidates
            if isinstance(item, type) and dataclasses.is_dataclass(item)
        ]
        if path in _NOT_TABLE_FIELDS:
            continue
        if nested:
            paths |= _result_fields(nested[0], f"{path}.")
        elif type_name.startswith("dict"):
            paths.add(f"{path}.<key>")
        else:
            paths.add(path)
    return paths


def _table_columns(table_type: str) -> set[str]:
    result_class, key_columns = _TABLE_TYPES[table_type]
    return key_columns | _result_fields(result_class)


def _handlers_section(start: str, end: str) -> str:
    page = _HANDLERS.read_text()
    return page[page.index(start) : page.index(end)]


def _handlers_column_dicts() -> dict[str, Any]:
    section = _handlers_section("#### Column dicts to copy", "#### Example")
    namespace: dict[str, Any] = {}
    for block in _PYTHON_BLOCK.findall(section):
        exec(textwrap.dedent(block), namespace)  # ruff: ignore[exec-builtin]
    return namespace


@pytest.mark.parametrize(
    ("table_type", "start", "end"),
    [
        (
            "functions",
            '??? info "Available fields for function tables"',
            '??? info "Available fields for gradient tables"',
        ),
        (
            "gradients",
            '??? info "Available fields for gradient tables"',
            "The `names` and",
        ),
    ],
)
def test_handlers_page_tabulates_every_result_field(
    table_type: str, start: str, end: str
) -> None:
    tabulated = set(_TABLE_FIELD.findall(_handlers_section(start, end)))
    expected = _table_columns(table_type)
    assert sorted(expected - tabulated) == [], "missing from the table"
    assert sorted(tabulated - expected) == [], "not a field of the results"


@pytest.mark.parametrize(
    ("table_type", "name"),
    [("functions", "FUNCTION_COLUMNS"), ("gradients", "GRADIENT_COLUMNS")],
)
def test_handlers_page_column_dicts_list_every_result_field(
    table_type: str, name: str
) -> None:
    section = _handlers_section("#### Column dicts to copy", "#### Example")
    block = next(item for item in _PYTHON_BLOCK.findall(section) if name in item)
    listed = set(_handlers_column_dicts()[name]) | set(_COMMENTED_FIELD.findall(block))
    expected = _table_columns(table_type)
    assert sorted(expected - listed) == [], "missing from the dict"
    assert sorted(listed - expected) == [], "not a field of the results"


def test_handlers_page_column_dicts_fill_tables() -> None:
    pytest.importorskip("polars")
    columns = _handlers_column_dicts()
    config = {
        "variables": {
            "variable_count": 2,
            "perturbation_magnitudes": 1e-6,
            "lower_bounds": -2.0,
            "upper_bounds": 2.0,
        },
        "realizations": {"weights": [1.0, 1.0]},
        "linear_constraints": {
            "coefficients": [[1.0, 1.0]],
            "lower_bounds": -3.0,
            "upper_bounds": 3.0,
        },
        "nonlinear_constraints": {"lower_bounds": -np.inf, "upper_bounds": 0.5},
        "optimizer": {"max_functions": 2},
    }
    tables = DataFrameHandler()
    tables.add_table("functions", "functions", columns["FUNCTION_COLUMNS"])
    tables.add_table("gradients", "gradients", columns["GRADIENT_COLUMNS"])
    optimize(
        config,
        np.zeros(2),
        lambda variables, _: [float(np.sum(variables**2)), float(variables[0])],
        handlers=[tables],
    )
    assert tables["functions"].height > 0
    assert tables["gradients"].height > 0
