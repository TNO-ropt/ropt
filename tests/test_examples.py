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


def test_rosenbrock_deterministic(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    module = _load_from_file("rosenbrock_deterministic")
    module.main()


@pytest.mark.parametrize("workflow", [True, False])
@pytest.mark.parametrize("merge", [True, False])
@pytest.mark.parametrize("function", [True, False])
def test_rosenbrock_ensemble(
    tmp_path: Path, monkeypatch: Any, workflow: Any, merge: Any, function: Any
) -> None:
    monkeypatch.chdir(tmp_path)
    module = _load_from_file("rosenbrock")
    module.main(workflow=workflow, merge=merge, function=function)


@pytest.mark.asyncio
@pytest.mark.parametrize("multiprocessing", [True, False])
async def test_rosenbrock_async(
    tmp_path: Path, monkeypatch: Any, multiprocessing: Any
) -> None:
    monkeypatch.chdir(tmp_path)

    # We need to do an explicit import, otherwise we get pickling errors:
    monkeypatch.syspath_prepend(Path(__file__).parent.parent / "examples")
    import rosenbrock_async  # type: ignore[import-not-found] # noqa: PLC0415

    await rosenbrock_async.main(multiprocessing=multiprocessing)


@pytest.mark.parametrize("linear", [True, False])
def test_differential_evolution(tmp_path: Path, monkeypatch: Any, linear: Any) -> None:
    monkeypatch.chdir(tmp_path)
    module = _load_from_file("differential_evolution")
    module.main(linear=linear)
