import dataclasses

import pytest

from ropt.enums import ExitCode
from ropt.exit_info import (
    _DEFAULT_MESSAGES,
    AbortFromErrorInfo,
    ExitInfo,
    MaxBatchesReachedInfo,
    MaxFunctionsReachedInfo,
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
