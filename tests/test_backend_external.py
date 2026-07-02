"""Tests for the external backend wrapper."""

from __future__ import annotations

import multiprocessing
from typing import cast

import numpy as np
import pytest

from ropt.config import BackendConfig
from ropt.context import EnOptContext
from ropt.enums import ExitCode
from ropt.exceptions import Abort
from ropt.exit_info import ExitInfo

cloudpickle = pytest.importorskip("cloudpickle")

from ropt.backend.external import (  # noqa: E402
    _decode_child_exception,
    _encode_child_exception,
    _run,
    _wrap_with_traceback,
)
from ropt.backend.scipy import SciPyBackend  # noqa: E402


def _make_child_args() -> bytes:
    context = EnOptContext.model_validate(
        {"variables": {"variable_count": 2, "perturbation_magnitudes": 1e-6}}
    )
    config = BackendConfig.model_validate({"method": "scipy/slsqp"})
    return cast(
        "bytes",
        cloudpickle.dumps(
            {
                "config": config,
                "context": context,
                "initial_values": np.zeros(2),
            }
        ),
    )


def test_child_abort_forwards_exit_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_abort(_self: SciPyBackend, _initial_values: np.ndarray) -> None:
        raise Abort(ExitInfo(exit_code=ExitCode.MAX_FUNCTIONS_REACHED))

    monkeypatch.setattr(SciPyBackend, "start", _raise_abort)

    ctx = multiprocessing.get_context("spawn")
    request_queue = ctx.Queue()
    result_queue = ctx.Queue()

    _run(_make_child_args(), request_queue, result_queue)

    abort_msg = request_queue.get(timeout=5)
    sentinel = request_queue.get(timeout=5)

    assert abort_msg["abort"] is True
    assert isinstance(abort_msg["info"], ExitInfo)
    assert abort_msg["info"].exit_code == ExitCode.MAX_FUNCTIONS_REACHED
    assert sentinel is None


def test_child_exception_is_cloudpickled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_value_error(_self: SciPyBackend, _initial_values: np.ndarray) -> None:
        msg = "delegate failed in child"
        raise ValueError(msg)

    monkeypatch.setattr(SciPyBackend, "start", _raise_value_error)

    ctx = multiprocessing.get_context("spawn")
    request_queue = ctx.Queue()
    result_queue = ctx.Queue()

    _run(_make_child_args(), request_queue, result_queue)

    payload = request_queue.get(timeout=5)
    sentinel = request_queue.get(timeout=5)

    assert "exception" in payload
    assert "traceback" in payload
    assert "delegate failed in child" in payload["traceback"]
    assert sentinel is None

    decoded = _decode_child_exception(payload)
    assert type(decoded) is ValueError
    assert str(decoded) == "delegate failed in child"
    assert any("delegate failed in child" in note for note in decoded.__notes__)


def test_unpicklable_child_exception_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _UnpicklableError(ValueError):
        pass

    def _raise_unpicklable(_self: SciPyBackend, _initial_values: np.ndarray) -> None:
        msg = "error cannot be pickled"
        raise _UnpicklableError(msg)

    monkeypatch.setattr(SciPyBackend, "start", _raise_unpicklable)

    # Build child args *before* breaking cloudpickle.dumps, otherwise
    # _make_child_args itself would fail.
    data = _make_child_args()

    real_dumps = cloudpickle.dumps

    def _failing_dumps(obj: object) -> bytes:
        if isinstance(obj, BaseException):
            msg = "cannot pickle"
            raise TypeError(msg)
        return cast("bytes", real_dumps(obj))

    monkeypatch.setattr("ropt.backend.external.cloudpickle.dumps", _failing_dumps)

    ctx = multiprocessing.get_context("spawn")
    request_queue = ctx.Queue()
    result_queue = ctx.Queue()

    _run(data, request_queue, result_queue)

    payload = request_queue.get(timeout=5)
    sentinel = request_queue.get(timeout=5)

    assert "exception" not in payload
    assert payload["error"] == "_UnpicklableError"
    assert payload["message"] == "error cannot be pickled"
    assert sentinel is None


def test_decode_falls_back_when_unpickle_fails() -> None:
    wrapper = _decode_child_exception(
        {"exception": b"\x00not a pickle", "traceback": "tb"}
    )

    assert type(wrapper) is RuntimeError
    assert wrapper.__cause__ is None
    assert any("tb" in note for note in wrapper.__notes__)


def test_decode_falls_back_for_non_exception_payload() -> None:
    """A pickled non-`Exception` payload triggers the fallback."""
    result = _decode_child_exception(
        {"exception": cloudpickle.dumps("not an exception"), "traceback": "tb"}
    )

    assert type(result) is RuntimeError
    assert "str" in str(result)


def test_encode_decode_round_trip() -> None:
    try:
        msg = "inner failure"
        raise RuntimeError(msg)  # noqa: TRY301
    except RuntimeError as exc:
        payload = _encode_child_exception(exc)

    assert "exception" in payload
    decoded = _decode_child_exception(payload)
    assert type(decoded) is RuntimeError
    assert str(decoded) == "inner failure"
    assert any("inner failure" in note for note in decoded.__notes__)


def test_wrap_with_traceback_attaches_note() -> None:
    wrapper = _wrap_with_traceback(
        "External backend subprocess raised ValueError: bad value", "child tb"
    )
    assert type(wrapper) is RuntimeError
    assert "ValueError" in str(wrapper)
    assert "bad value" in str(wrapper)
    assert any("child tb" in note for note in wrapper.__notes__)
