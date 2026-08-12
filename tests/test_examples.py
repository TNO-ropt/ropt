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

    # We need to do an explicit import, otherwise we get pickling errors:
    monkeypatch.syspath_prepend(Path(__file__).parent.parent / "examples" / "advanced")
    import parallel_evaluator  # type: ignore[import-not-found] # ruff: ignore[import-outside-top-level]

    await parallel_evaluator.main(multiprocessing=multiprocessing)


@pytest.mark.slow
def test_example_nested(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    module = _load_from_file("nested")
    module.main()


@pytest.mark.slow
def test_example_nested_multiprocess(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)

    # We need to do an explicit import, otherwise we get pickling errors:
    monkeypatch.syspath_prepend(Path(__file__).parent.parent / "examples" / "advanced")
    import nested_multiprocess  # type: ignore[import-not-found] # ruff: ignore[import-outside-top-level]

    nested_multiprocess.main()


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
