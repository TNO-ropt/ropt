import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

_EXAMPLES = Path(__file__).parent.parent / "examples"


@pytest.fixture(autouse=True)
def _examples_importable(monkeypatch: Any) -> None:
    monkeypatch.syspath_prepend(str(_EXAMPLES))
    monkeypatch.setenv("PYTHONPATH", str(_EXAMPLES))


def _load_from_file(name: str) -> Any:
    assert (_EXAMPLES / f"{name}.py").exists()
    sys.modules.pop(name, None)
    return importlib.import_module(name)


def test_example_evaluate(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _load_from_file("evaluate").main()


def test_example_parallel_threads(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _load_from_file("parallel").main(multiprocessing=False)


def test_example_multiple_optimizations(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _load_from_file("multiple_optimizations").main()


def test_example_metadata(tmp_path: Path, monkeypatch: Any) -> None:
    pytest.importorskip("polars")
    monkeypatch.chdir(tmp_path)
    _load_from_file("metadata").main()


def test_example_callback_metadata(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _load_from_file("callback_metadata").main()


def test_example_export_polars(tmp_path: Path, monkeypatch: Any) -> None:
    pytest.importorskip("polars")
    monkeypatch.chdir(tmp_path)
    _load_from_file("export").main()


def test_example_export_pandas(tmp_path: Path, monkeypatch: Any) -> None:
    pytest.importorskip("pandas")
    monkeypatch.chdir(tmp_path)
    _load_from_file("export").main(pandas=True)


def test_example_failures(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _load_from_file("failures").main()


def test_example_stopping(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _load_from_file("stopping").main()


def test_example_handlers(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _load_from_file("handlers").main()


def test_example_tables(tmp_path: Path, monkeypatch: Any) -> None:
    pytest.importorskip("polars")
    monkeypatch.chdir(tmp_path)
    _load_from_file("tables").main()


def test_example_restart(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _load_from_file("restart").main()


def test_example_initial_values(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _load_from_file("initial_values").main()


@pytest.mark.slow
def test_example_nested_optimization(tmp_path: Path, monkeypatch: Any) -> None:
    pytest.importorskip("polars")
    monkeypatch.chdir(tmp_path)
    _load_from_file("nested_optimization").main()


def test_example_ensemble(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _load_from_file("ensemble").main()


def test_example_realization_filter(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _load_from_file("realization_filter").main()


def test_example_function_estimator(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _load_from_file("function_estimator").main()


def test_example_sampler(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _load_from_file("sampler").main()


def test_example_scaling(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _load_from_file("scaling").main()


@pytest.mark.parametrize("linear", [True, False])
def test_example_constrained(tmp_path: Path, monkeypatch: Any, linear: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _load_from_file("constrained").main(linear=linear)


@pytest.mark.parametrize("linear", [True, False])
def test_example_discrete(tmp_path: Path, monkeypatch: Any, linear: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _load_from_file("discrete").main(linear=linear)


@pytest.mark.slow
def test_example_mixed(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _load_from_file("mixed").main()


@pytest.mark.slow
def test_example_hpc_on_the_local_queue(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _load_from_file("hpc").main(queue="local", workdir=tmp_path)


@pytest.mark.slow
def test_example_programs(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _load_from_file("programs").main()
