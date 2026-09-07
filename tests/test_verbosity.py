"""Tests for the verbosity setting."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from ropt.backend.utils import resolve_verbosity
from ropt.components.evaluators import (
    EvaluationFunctionContext,
    EvaluationFunctionResult,
)
from ropt.config import BackendConfig
from ropt.simple import optimize

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


@pytest.mark.parametrize(
    ("verbose", "expected"),
    [
        (None, 0),
        (False, 0),
        (True, None),
        (0, 0),
        (1, 1),
        (3, 3),
    ],
)
def test_resolve_verbosity_maps_true_to_the_default_level_and_false_to_silence(
    *, verbose: bool | int | None, expected: int | None
) -> None:
    assert resolve_verbosity(verbose=verbose) == expected


def test_resolve_verbosity_keeps_true_distinct_from_one() -> None:
    assert resolve_verbosity(verbose=True) is None
    assert resolve_verbosity(verbose=1) == 1


def test_backend_config_rejects_a_negative_verbosity() -> None:
    with pytest.raises(ValueError, match="verbose"):
        BackendConfig(verbose=-1)


def test_backend_config_keeps_booleans_and_integers_apart() -> None:
    assert BackendConfig(verbose=True).verbose is True
    assert BackendConfig(verbose=1).verbose == 1
    assert not isinstance(BackendConfig(verbose=1).verbose, bool)


def _objective(
    variables: NDArray[np.float64],
    _context: EvaluationFunctionContext,
) -> EvaluationFunctionResult:
    return EvaluationFunctionResult(objectives=np.array([float(np.sum(variables**2))]))


def _config(**backend: Any) -> dict[str, Any]:
    return {
        "optimizer": {"max_batches": 10},
        "variables": {"variable_count": 2, "perturbation_magnitudes": 0.01},
        "objectives": {"weights": [1.0]},
        "backend": {"method": "scipy/slsqp", "max_iterations": 2, **backend},
    }


def test_scipy_is_silent_unless_verbosity_is_requested(
    capsys: pytest.CaptureFixture[str],
) -> None:
    optimize(_config(), np.ones(2), _objective)
    assert not capsys.readouterr().out


def test_scipy_reports_when_verbosity_is_requested(
    capsys: pytest.CaptureFixture[str],
) -> None:
    optimize(_config(verbose=True), np.ones(2), _objective)
    assert "Current function value" in capsys.readouterr().out


def test_scipy_stays_silent_when_only_the_output_destination_is_set(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Capture decides where output goes, not whether there is any.
    log = tmp_path / "run.log"
    optimize(
        {**_config(), "optimizer": {"max_batches": 10, "stdout": log}},
        np.ones(2),
        _objective,
    )
    assert not log.read_text()
    assert not capsys.readouterr().out


def test_verbose_output_goes_to_the_configured_file(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    optimize(
        {
            **_config(verbose=True),
            "optimizer": {"max_batches": 10, "stdout": log},
        },
        np.ones(2),
        _objective,
    )
    assert "Current function value" in log.read_text()


def test_a_backend_option_overrides_the_verbosity_setting(
    capsys: pytest.CaptureFixture[str],
) -> None:
    optimize(_config(verbose=False, options={"disp": True}), np.ones(2), _objective)
    assert "Current function value" in capsys.readouterr().out
