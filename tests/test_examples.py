from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any


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


def test_rosenbrock_ensemble(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    module = _load_from_file("rosenbrock_ensemble")
    module.main()


def test_rosenbrock_ensemble_merged(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    module = _load_from_file("rosenbrock_ensemble_merged")
    module.main()


def test_rosenbrock_workflow(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    module = _load_from_file("rosenbrock_workflow")
    module.main()


def test_differential_evolution_linear(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    module = _load_from_file("de_linear")
    module.main()


def test_differential_evolution_nonlinear(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    module = _load_from_file("de_nonlinear")
    module.main()
