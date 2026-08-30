import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

_EXAMPLES = Path(__file__).parent.parent / "examples"


@pytest.fixture(autouse=True)
def _examples_importable(monkeypatch: Any) -> None:
    # Autouse because every test in this module loads an example by bare name,
    # so requesting it explicitly would add an unused argument to all of them.
    # Run as scripts, the examples' functions live in `__main__`, which `spawn`
    # re-imports, so a worker rebuilds them by name and no example needs
    # `cloudpickle`. Executing the file here under a name nothing can import
    # would break that and make the tests demand an extra the examples do not.
    for sub_path in ("advanced", "simple"):
        monkeypatch.syspath_prepend(str(_EXAMPLES / sub_path))


def _load_from_file(name: str, sub_path: str = "advanced") -> Any:
    assert (_EXAMPLES / sub_path / f"{name}.py").exists()
    # Popped rather than reused: a parametrized test loads the same example
    # again, and it should start from the top as a script would.
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
