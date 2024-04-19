import importlib
import os
import shutil
from pathlib import Path
from typing import Any, Optional

import pytest


def _load_from_file(name: str, sub_path: Optional[str] = None) -> Any:
    path = Path(__file__).parent.parent / "examples"
    if sub_path is not None:
        path = path / sub_path
    path = path / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_rosenbrock_deterministic(tmp_path: Path) -> None:
    os.chdir(tmp_path)
    module = _load_from_file("rosenbrock_deterministic")
    module.main()


def test_rosenbrock_ensemble(tmp_path: Path) -> None:
    os.chdir(tmp_path)
    module = _load_from_file("rosenbrock_ensemble")
    module.main()


def test_rosenbrock_ensemble_merged(tmp_path: Path) -> None:
    os.chdir(tmp_path)
    module = _load_from_file("rosenbrock_ensemble")
    module.main(["--merge"])


def test_differential_evolution_linear(tmp_path: Path) -> None:
    os.chdir(tmp_path)
    module = _load_from_file("de_linear")
    module.main()


def test_differential_evolution_nonlinear(tmp_path: Path) -> None:
    os.chdir(tmp_path)
    module = _load_from_file("de_nonlinear")
    module.main()


@pytest.mark.slow()
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_script_optimizer(tmp_path: Path) -> None:
    os.chdir(tmp_path)
    module = _load_from_file("run", sub_path="script_optimizer")
    shutil.copyfile(
        Path(__file__).parent.parent
        / "examples"
        / "script_optimizer"
        / "rosenbrock.py",
        tmp_path / "rosenbrock.py",
    )
    module.main()
