from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import pytest


def _load_from_file(name: str, sub_path: str | None = None) -> Any:
    path = Path(__file__).parent.parent / "examples"
    if sub_path is not None:
        path /= sub_path
    path /= f"{name}.py"
    spec = spec_from_file_location(name, path)
    assert spec is not None
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_example_deterministic(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    module = _load_from_file("deterministic")
    module.main()


@pytest.mark.parametrize("merge", [True, False])
def test_example_ensemble(tmp_path: Path, monkeypatch: Any, merge: Any) -> None:
    monkeypatch.chdir(tmp_path)
    module = _load_from_file("ensemble")
    module.main(merge=merge)


@pytest.mark.parametrize("merge", [True, False])
def test_example_function_evaluator(
    tmp_path: Path, monkeypatch: Any, merge: Any
) -> None:
    monkeypatch.chdir(tmp_path)
    module = _load_from_file("function_evaluator")
    module.main(merge=merge)


@pytest.mark.parametrize("merge", [True, False])
def test_example_workflow(tmp_path: Path, monkeypatch: Any, merge: Any) -> None:
    monkeypatch.chdir(tmp_path)
    module = _load_from_file("workflow")
    module.main(merge=merge)


@pytest.mark.parametrize("linear", [True, False])
def test_example_constrained(tmp_path: Path, monkeypatch: Any, linear: Any) -> None:
    monkeypatch.chdir(tmp_path)
    module = _load_from_file("constrained")
    module.main(linear=linear)


@pytest.mark.slow
@pytest.mark.asyncio
@pytest.mark.parametrize("multiprocessing", [True, False])
async def test_example_async_evaluator(
    tmp_path: Path, monkeypatch: Any, multiprocessing: Any
) -> None:
    monkeypatch.chdir(tmp_path)

    # We need to do an explicit import, otherwise we get pickling errors:
    monkeypatch.syspath_prepend(Path(__file__).parent.parent / "examples")
    import async_evaluator  # type: ignore[import-not-found] # noqa: PLC0415

    await async_evaluator.main(multiprocessing=multiprocessing)


@pytest.mark.slow
def test_example_discrete(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    module = _load_from_file("discrete")
    module.main()


@pytest.mark.slow
def test_example_nested(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    module = _load_from_file("nested")
    module.main()


@pytest.mark.slow
def test_example_nested_multiprocess(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)

    # We need to do an explicit import, otherwise we get pickling errors:
    monkeypatch.syspath_prepend(Path(__file__).parent.parent / "examples")
    import nested_multiprocess  # type: ignore[import-not-found] # noqa: PLC0415

    nested_multiprocess.main()


@pytest.mark.parametrize("linear", [True, False])
def test_example_differential_evolution(
    tmp_path: Path, monkeypatch: Any, linear: Any
) -> None:
    monkeypatch.chdir(tmp_path)
    module = _load_from_file("differential_evolution")
    module.main(linear=linear)
