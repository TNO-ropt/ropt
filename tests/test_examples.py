from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import pytest


def _load_from_file(name: str, sub_path: str = "advanced") -> Any:
    path = Path(__file__).parent.parent / "examples" / sub_path / f"{name}.py"
    spec = spec_from_file_location(name, path)
    assert spec is not None
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("merge", [True, False])
def test_example_workflow(tmp_path: Path, monkeypatch: Any, merge: Any) -> None:
    monkeypatch.chdir(tmp_path)
    module = _load_from_file("workflow")
    module.main(merge=merge)


@pytest.mark.slow
@pytest.mark.asyncio
@pytest.mark.parametrize("multiprocessing", [True, False])
async def test_example_parallel_evaluator(
    tmp_path: Path, monkeypatch: Any, multiprocessing: Any
) -> None:
    monkeypatch.chdir(tmp_path)
    module = _load_from_file("parallel_evaluator")
    await module.main(multiprocessing=multiprocessing)


@pytest.mark.slow
def test_example_nested(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    module = _load_from_file("nested")
    module.main()


@pytest.mark.slow
def test_example_nested_multiprocess(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    module = _load_from_file("nested_multiprocess")
    module.main()


def test_example_simple_optimize(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _load_from_file("optimize", "simple").main()


def test_example_simple_evaluate(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _load_from_file("evaluate", "simple").main()


def test_example_simple_parallel_threads(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _load_from_file("parallel", "simple").main(multiprocessing=False)


def test_example_simple_optimize_many(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _load_from_file("optimize_many", "simple").main()


def test_example_simple_metadata(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _load_from_file("metadata", "simple").main()


def test_example_simple_handlers(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _load_from_file("handlers", "simple").main()


@pytest.mark.slow
def test_example_simple_nested_optimization(tmp_path: Path, monkeypatch: Any) -> None:
    pytest.importorskip("polars")
    # Not the example's own requirement: run as a script its functions live in
    # `__main__`, which spawn re-imports. Loading it here under a name the
    # worker cannot import is what defeats pickle-by-reference.
    pytest.importorskip("cloudpickle")
    monkeypatch.chdir(tmp_path)
    _load_from_file("nested_optimization", "simple").main()


@pytest.mark.parametrize("merge", [True, False])
def test_example_simple_ensemble(tmp_path: Path, monkeypatch: Any, merge: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _load_from_file("ensemble", "simple").main(merge=merge)


@pytest.mark.parametrize("linear", [True, False])
def test_example_simple_constrained(
    tmp_path: Path, monkeypatch: Any, linear: Any
) -> None:
    monkeypatch.chdir(tmp_path)
    _load_from_file("constrained", "simple").main(linear=linear)


@pytest.mark.parametrize("linear", [True, False])
def test_example_simple_discrete(tmp_path: Path, monkeypatch: Any, linear: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _load_from_file("discrete", "simple").main(linear=linear)


@pytest.mark.slow
def test_example_simple_mixed(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _load_from_file("mixed", "simple").main()
