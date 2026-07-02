import dataclasses

import pytest

from ropt.enums import ExitCode
from ropt.exit_info import (
    _DEFAULT_MESSAGES,
    AbortFromErrorInfo,
    ExitInfo,
    MaxBatchesReachedInfo,
    MaxFunctionsReachedInfo,
    TooFewRealizationsInfo,
)


def test_exit_info_default_message() -> None:
    info = ExitInfo(exit_code=ExitCode.OPTIMIZER_FINISHED)
    assert info.message == _DEFAULT_MESSAGES[ExitCode.OPTIMIZER_FINISHED]


def test_exit_info_explicit_message() -> None:
    info = ExitInfo(exit_code=ExitCode.TOO_FEW_REALIZATIONS, message="No survivors")
    assert info.message == "No survivors"


def test_exit_info_default_message_for_all_exit_codes() -> None:
    for exit_code in ExitCode:
        info = ExitInfo(exit_code=exit_code)
        assert info.message


def test_exit_info_frozen() -> None:
    info = ExitInfo(exit_code=ExitCode.OPTIMIZER_FINISHED)
    with pytest.raises(dataclasses.FrozenInstanceError):
        info.exit_code = ExitCode.USER_ABORT  # type: ignore[misc]


def test_max_functions_reached_info() -> None:
    info = MaxFunctionsReachedInfo(limit=5)
    assert info.exit_code == ExitCode.MAX_FUNCTIONS_REACHED
    assert info.limit == 5


def test_max_functions_reached_info_message_includes_limit() -> None:
    info = MaxFunctionsReachedInfo(limit=7)
    assert "7" in info.message


def test_max_functions_reached_info_explicit_message() -> None:
    info = MaxFunctionsReachedInfo(limit=7, message="custom")
    assert info.message == "custom"


def test_max_batches_reached_info() -> None:
    info = MaxBatchesReachedInfo(limit=3)
    assert info.exit_code == ExitCode.MAX_BATCHES_REACHED
    assert info.limit == 3


def test_max_batches_reached_info_message_includes_limit() -> None:
    info = MaxBatchesReachedInfo(limit=42)
    assert "42" in info.message


def test_abort_from_error_info() -> None:
    info = AbortFromErrorInfo(error="boom")
    assert info.exit_code == ExitCode.ABORT_FROM_ERROR
    assert info.error == "boom"


def test_abort_from_error_info_message_includes_error() -> None:
    info = AbortFromErrorInfo(error="disk full")
    assert "disk full" in info.message


def test_too_few_realizations_info_function_only() -> None:
    info = TooFewRealizationsInfo(
        failed_functions=2,
        failed_gradients=0,
        failed_perturbations=0,
        realization_min_success=3,
    )
    assert info.exit_code == ExitCode.TOO_FEW_REALIZATIONS
    assert info.failed_functions == 2
    assert info.failed_gradients == 0
    assert info.failed_perturbations == 0
    assert info.realization_min_success == 3
    assert info.perturbation_min_success is None
    assert "2 function result(s)" in info.message
    assert "gradient" not in info.message
    assert "3 successful realization(s)" in info.message
    assert "perturbation" not in info.message


def test_too_few_realizations_info_gradient_only() -> None:
    info = TooFewRealizationsInfo(
        failed_functions=0,
        failed_gradients=1,
        failed_perturbations=0,
        realization_min_success=2,
        perturbation_min_success=4,
    )
    assert "1 gradient result(s)" in info.message
    assert "function" not in info.message
    assert "perturbation" not in info.message


def test_too_few_realizations_info_mixed_batch() -> None:
    info = TooFewRealizationsInfo(
        failed_functions=1,
        failed_gradients=2,
        failed_perturbations=0,
        realization_min_success=2,
        perturbation_min_success=1,
    )
    assert "1 function result(s)" in info.message
    assert "2 gradient result(s)" in info.message
    assert "and" in info.message


def test_too_few_realizations_info_perturbation_addendum() -> None:
    info = TooFewRealizationsInfo(
        failed_functions=0,
        failed_gradients=2,
        failed_perturbations=1,
        realization_min_success=3,
        perturbation_min_success=5,
    )
    assert "2 gradient result(s)" in info.message
    assert (
        "1 of the failed gradient result(s) had realization(s) that"
        " fell below the minimum of 5 successful perturbation(s)."
    ) in info.message


def test_too_few_realizations_info_explicit_message() -> None:
    info = TooFewRealizationsInfo(
        failed_functions=1,
        failed_gradients=0,
        failed_perturbations=0,
        realization_min_success=2,
        message="custom",
    )
    assert info.message == "custom"
