import importlib
import os
import sys
from pathlib import Path
from typing import Any

import pytest

_EXAMPLES = Path(__file__).parent.parent / "examples"


@pytest.fixture(autouse=True)
def _examples_importable(monkeypatch: Any) -> None:
    for sub_path in ("advanced", "simple"):
        monkeypatch.syspath_prepend(str(_EXAMPLES / sub_path))
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(str(_EXAMPLES / p) for p in ("advanced", "simple")),
    )


def _load_from_file(name: str, sub_path: str = "advanced") -> Any:
    assert (_EXAMPLES / sub_path / f"{name}.py").exists()
    sys.modules.pop(name, None)
    return importlib.import_module(name)


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
def test_example_nested_parallel(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    module = _load_from_file("nested_parallel")
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


def test_example_simple_restart(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _load_from_file("restart", "simple").main()


@pytest.mark.slow
def test_example_simple_nested_optimization(tmp_path: Path, monkeypatch: Any) -> None:
    pytest.importorskip("polars")
    monkeypatch.chdir(tmp_path)
    _load_from_file("nested_optimization", "simple").main()


@pytest.mark.parametrize("merge", [True, False])
def test_example_simple_ensemble(tmp_path: Path, monkeypatch: Any, merge: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _load_from_file("ensemble", "simple").main(merge=merge)


def test_example_simple_realization_filter(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _load_from_file("realization_filter", "simple").main()


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


@pytest.mark.slow
def test_example_simple_hpc_on_a_local_pool(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _load_from_file("hpc", "simple").main(local=True, workdir=tmp_path)


@pytest.mark.slow
def test_example_advanced_hpc_on_a_local_pool(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _load_from_file("hpc_executor").main(workdir=tmp_path, local=True)
