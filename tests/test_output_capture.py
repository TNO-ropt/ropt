"""Tests for capturing optimizer output."""

# The prints and the subprocess are what these tests exercise.
# ruff: file-ignore[print, suspicious-subprocess-import]
# ruff: file-ignore[subprocess-without-shell-equals-true]

from __future__ import annotations

import ctypes
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pytest

from ropt.backend import Backend
from ropt.backend.utils import collect_native_output
from ropt.components.evaluators import (
    EvaluationFunctionContext,
    EvaluationFunctionResult,
)
from ropt.exceptions import WorkflowError
from ropt.plugins.manager import register_plugin
from ropt.simple import optimize

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ropt.config import BackendConfig
    from ropt.context import EnOptContext
    from ropt.core import OptimizerCallback
    from ropt.plugins import MethodSpec

_libc = ctypes.CDLL(None)


def _native_print(text: str) -> None:
    _libc.printf(b"%s", text.encode())


class _PrintingBackend(Backend):
    """Prints at both levels, and runs one evaluation in between."""

    methods: ClassVar[MethodSpec] = {"printing"}

    def __init__(self, backend_config: BackendConfig) -> None:
        self._config = backend_config

    def init(
        self, context: EnOptContext, optimizer_callback: OptimizerCallback
    ) -> None:
        self._context = context
        self._callback = optimizer_callback

    def start(self, initial_values: NDArray[np.float64]) -> None:
        print("OPTIMIZER-PYTHON-STDOUT")
        print("OPTIMIZER-PYTHON-STDERR", file=sys.stderr)
        _native_print("OPTIMIZER-NATIVE\n")
        self._callback(initial_values, return_functions=True, return_gradients=False)
        print("OPTIMIZER-AFTER-EVALUATION")

    def validate_options(self) -> None:
        pass


class _NativePrintingBackend(_PrintingBackend):
    methods: ClassVar[MethodSpec] = {"printing"}

    @property
    def bypasses_python_output(self) -> bool:
        return True


@pytest.fixture(autouse=True)
def _backends() -> None:
    register_plugin("backend", "printing", _PrintingBackend)
    register_plugin("backend", "native", _NativePrintingBackend)


def _objective(
    variables: NDArray[np.float64],
    _context: EvaluationFunctionContext,
) -> EvaluationFunctionResult:
    print("CALLBACK-OUTPUT")
    return EvaluationFunctionResult(objectives=np.array([float(np.sum(variables**2))]))


def _config(method: str, **optimizer: Any) -> dict[str, Any]:
    return {
        "optimizer": {"max_batches": 1, **optimizer},
        "variables": {"variable_count": 2, "perturbation_magnitudes": 0.01},
        "objectives": {"weights": [1.0]},
        "backend": {"method": method},
    }


def _run(method: str, **optimizer: Any) -> None:
    optimize(_config(method, **optimizer), np.ones(2), _objective)


def test_capture_writes_optimizer_output_to_the_configured_file(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    _run("printing/printing", stdout=log)
    assert "OPTIMIZER-PYTHON-STDOUT" in log.read_text()
    assert "OPTIMIZER-AFTER-EVALUATION" in log.read_text()


def test_capture_leaves_evaluation_output_on_the_original_stream(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = tmp_path / "run.log"
    _run("printing/printing", stdout=log)
    assert "CALLBACK-OUTPUT" in capsys.readouterr().out
    assert "CALLBACK-OUTPUT" not in log.read_text()


def test_capture_sends_stderr_to_the_stdout_file_when_only_stdout_is_set(
    tmp_path: Path,
) -> None:
    log = tmp_path / "run.log"
    _run("printing/printing", stdout=log)
    captured = log.read_text()
    assert "OPTIMIZER-PYTHON-STDOUT" in captured
    assert "OPTIMIZER-PYTHON-STDERR" in captured


def test_capture_honors_stderr_configured_without_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    errors = tmp_path / "errors.log"
    _run("printing/printing", stderr=errors)
    assert "OPTIMIZER-PYTHON-STDERR" in errors.read_text()
    assert "OPTIMIZER-PYTHON-STDOUT" in capsys.readouterr().out


def test_capture_keeps_both_streams_when_they_share_one_file(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    _run("printing/printing", stdout=log, stderr=log)
    captured = log.read_text()
    assert "OPTIMIZER-PYTHON-STDOUT" in captured
    assert "OPTIMIZER-PYTHON-STDERR" in captured


def test_capture_resolves_a_relative_path_against_the_output_directory(
    tmp_path: Path,
) -> None:
    _run("printing/printing", stdout="run.log", output_dir=tmp_path)
    assert "OPTIMIZER-PYTHON-STDOUT" in (tmp_path / "run.log").read_text()


def test_capture_truncates_an_existing_file(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text("STALE-CONTENT-FROM-A-PREVIOUS-RUN\n" * 100)
    _run("printing/printing", stdout=log)
    assert "STALE-CONTENT-FROM-A-PREVIOUS-RUN" not in log.read_text()


def test_capture_creates_files_without_execute_permission(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    _run("printing/printing", stdout=log)
    assert not log.stat().st_mode & 0o111


def test_capture_ignores_native_output_when_the_backend_does_not_declare_it(
    tmp_path: Path,
) -> None:
    log = tmp_path / "run.log"
    _run("printing/printing", stdout=log)
    assert "OPTIMIZER-NATIVE" not in log.read_text()


def test_capture_collects_native_output_when_the_backend_declares_it(
    tmp_path: Path,
) -> None:
    log = tmp_path / "run.log"
    _run("native/printing", stdout=log)
    assert "OPTIMIZER-NATIVE" in log.read_text()


def test_capture_refuses_a_second_run_while_one_is_active(tmp_path: Path) -> None:
    inner_log = tmp_path / "inner.log"

    def nested_objective(
        variables: NDArray[np.float64],
        _context: EvaluationFunctionContext,
    ) -> EvaluationFunctionResult:
        optimize(_config("printing/printing", stdout=inner_log), variables, _objective)
        return EvaluationFunctionResult(objectives=np.array([1.0]))

    with pytest.raises(WorkflowError, match="already being captured"):
        optimize(
            _config("printing/printing", stdout=tmp_path / "outer.log"),
            np.ones(2),
            nested_objective,
        )


def test_capture_allows_a_second_run_after_the_first_finishes(tmp_path: Path) -> None:
    _run("printing/printing", stdout=tmp_path / "first.log")
    _run("printing/printing", stdout=tmp_path / "second.log")
    assert "OPTIMIZER-PYTHON-STDOUT" in (tmp_path / "second.log").read_text()


def test_capture_releases_the_claim_when_the_run_fails(tmp_path: Path) -> None:
    def failing_objective(
        _variables: NDArray[np.float64],
        _context: EvaluationFunctionContext,
    ) -> EvaluationFunctionResult:
        msg = "objective failed"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="objective failed"):
        optimize(
            _config("printing/printing", stdout=tmp_path / "failed.log"),
            np.ones(2),
            failing_objective,
        )
    _run("printing/printing", stdout=tmp_path / "after.log")
    assert "OPTIMIZER-PYTHON-STDOUT" in (tmp_path / "after.log").read_text()


def test_capture_does_not_leak_file_descriptors(tmp_path: Path) -> None:
    descriptors = Path("/proc/self/fd")
    if not descriptors.is_dir():
        pytest.skip("descriptor counting requires /proc")
    before = len(list(descriptors.iterdir()))
    for index in range(5):
        _run("native/printing", stdout=tmp_path / f"run{index}.log")
    assert len(list(descriptors.iterdir())) == before


_SUBPROCESS_SCRIPT = """
import sys
import numpy as np
from ropt.components.evaluators import EvaluationFunctionResult
from ropt.plugins.manager import register_plugin
from ropt.simple import optimize
sys.path.insert(0, {tests!r})
from test_output_capture import _NativePrintingBackend, _objective

register_plugin("backend", "native", _NativePrintingBackend)
optimize(
    {{
        "optimizer": {{"max_batches": 1, "stdout": {log!r}}},
        "variables": {{"variable_count": 2, "perturbation_magnitudes": 0.01}},
        "objectives": {{"weights": [1.0]}},
        "backend": {{"method": "native/printing"}},
    }},
    np.ones(2),
    _objective,
)
"""


def test_capture_is_not_defeated_by_a_block_buffered_stdout(tmp_path: Path) -> None:
    # Piping stdout makes both Python and libc fully buffer it, so anything that
    # is not flushed at the boundaries drains after the redirect is undone.
    log = tmp_path / "run.log"
    script = _SUBPROCESS_SCRIPT.format(tests=str(Path(__file__).parent), log=str(log))
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        check=True,
    )
    captured = log.read_text()
    assert "OPTIMIZER-PYTHON-STDOUT" in captured
    assert "OPTIMIZER-NATIVE" in captured
    assert "OPTIMIZER-PYTHON-STDOUT" not in result.stdout
    assert "OPTIMIZER-NATIVE" not in result.stdout
    assert "CALLBACK-OUTPUT" in result.stdout


def test_collect_native_output_separates_the_two_levels() -> None:
    def run() -> None:
        print("PYTHON-LEVEL")
        _native_print("NATIVE-LEVEL\n")

    native = collect_native_output(run)
    assert "NATIVE-LEVEL" in native
    assert "PYTHON-LEVEL" not in native


def test_collect_native_output_reports_what_a_backend_writes_natively() -> None:
    native = collect_native_output(lambda: _run("printing/printing"))
    assert "OPTIMIZER-NATIVE" in native
    assert "OPTIMIZER-PYTHON-STDOUT" not in native
