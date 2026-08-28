"""Tests for the external backend wrapper."""

from __future__ import annotations

import multiprocessing
import pickle  # ruff: ignore[suspicious-pickle-import]
import sys
from typing import Any, cast

import numpy as np
import pytest

from ropt._serialize import (
    CANNOT_DESERIALIZE,
    CANNOT_SERIALIZE,
    HAVE_CLOUDPICKLE,
    dumps,
)
from ropt.backend.external import (
    ExternalBackend,
    _decode_child_exception,
    _encode_child_exception,
    _run,
    _wrap_with_traceback,
)
from ropt.backend.scipy import SciPyBackend
from ropt.components.evaluators import EvaluationFunctionResult, FunctionEvaluator
from ropt.config import BackendConfig
from ropt.context import EnOptContext
from ropt.enums import ExitCode
from ropt.exceptions import ExecutionError, OptimizerStop
from ropt.simple import optimize


def _make_context() -> EnOptContext:
    return EnOptContext.model_validate(
        {"variables": {"variable_count": 2, "perturbation_magnitudes": 1e-6}}
    )


def _make_child_args() -> bytes:
    config = BackendConfig.model_validate({"method": "scipy/slsqp"})
    return dumps(
        {
            "config": config,
            "context": _make_context(),
            "initial_values": np.zeros(2),
        }
    )


def test_child_abort_forwards_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_abort(_self: SciPyBackend, _initial_values: np.ndarray) -> None:
        raise OptimizerStop(ExitCode.MAX_FUNCTIONS_REACHED)

    monkeypatch.setattr(SciPyBackend, "start", _raise_abort)

    ctx = multiprocessing.get_context("spawn")
    request_queue = ctx.Queue()
    result_queue = ctx.Queue()

    _run(_make_child_args(), request_queue, result_queue)

    abort_msg = request_queue.get(timeout=5)
    sentinel = request_queue.get(timeout=5)

    assert abort_msg["stop"] is True
    assert abort_msg["exit_code"] == ExitCode.MAX_FUNCTIONS_REACHED
    assert sentinel is None


def test_child_exception_is_serialized(
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


def test_unserializable_child_exception_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _UnserializableError(ValueError):
        pass

    def _raise_unserializable(_self: SciPyBackend, _initial_values: np.ndarray) -> None:
        msg = "error cannot be pickled"
        raise _UnserializableError(msg)

    monkeypatch.setattr(SciPyBackend, "start", _raise_unserializable)

    # Build child args *before* breaking `dumps`, otherwise _make_child_args
    # itself would fail.
    data = _make_child_args()

    def _failing_dumps(obj: object) -> bytes:
        if isinstance(obj, BaseException):
            msg = "cannot pickle"
            raise TypeError(msg)
        return dumps(obj)

    monkeypatch.setattr("ropt.backend.external.dumps", _failing_dumps)

    ctx = multiprocessing.get_context("spawn")
    request_queue = ctx.Queue()
    result_queue = ctx.Queue()

    _run(data, request_queue, result_queue)

    payload = request_queue.get(timeout=5)
    sentinel = request_queue.get(timeout=5)

    assert "exception" not in payload
    assert payload["error"] == "_UnserializableError"
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
        {"exception": dumps("not an exception"), "traceback": "tb"}
    )

    assert type(result) is RuntimeError
    assert "str" in str(result)


def test_encode_decode_round_trip() -> None:
    try:
        msg = "inner failure"
        raise RuntimeError(msg)  # ruff: ignore[raise-within-try]
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


def _record_stdlib_dumps(monkeypatch: pytest.MonkeyPatch, target: str) -> list[Any]:
    """Patch `target` to the standard library's `dumps`, recording every call.

    The record is what makes the patch verifiable. A `cloudpickle` payload reads
    back with the standard library's unpickler, so a patch that quietly stopped
    landing would leave every other assertion in the test still passing.

    Args:
        monkeypatch: The fixture that installs and later removes the patch.
        target:      The `dumps` symbol to replace, named per module.

    Returns:
        The list the patched `dumps` appends each serialized object to.
    """
    calls: list[Any] = []

    def _dumps(obj: Any) -> bytes:
        calls.append(obj)
        return pickle.dumps(obj)

    monkeypatch.setattr(target, _dumps)
    return calls


def test_the_problem_travels_without_cloudpickle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`cloudpickle` is optional: the standard library must be able to send this.

    The evaluations stay in the parent process, so the objective is held by an
    evaluator, which serializes as an inert placeholder. It is therefore never
    sent, and what is sent can be looked up by name.
    """
    context = EnOptContext.model_validate(
        {
            "variables": {"variable_count": 2, "perturbation_magnitudes": 1e-6},
            "backend": {"method": "external/scipy/slsqp"},
        }
    )
    backend = context.backend
    assert isinstance(backend, ExternalBackend)

    # A closure, which only `cloudpickle` could send, reached the way a real
    # run reaches it: through an evaluator.
    target = 0.5
    evaluator = FunctionEvaluator(
        function=lambda variables, _: EvaluationFunctionResult(
            objectives=np.array([float(((variables - target) ** 2).sum())])
        )
    )
    backend.init(context, cast("Any", evaluator.eval))

    # The standard library, whether or not `cloudpickle` is installed.
    dumped = _record_stdlib_dumps(monkeypatch, "ropt.backend.external.dumps")
    restored = pickle.loads(backend._serialize(np.zeros(2)))  # ruff: ignore[private-member-access, suspicious-pickle-usage]

    assert dumped
    assert restored["config"].method == "scipy/slsqp"
    assert np.array_equal(restored["initial_values"], np.zeros(2))


class _VanishingMarker:
    """Pickles by name here, and does not resolve once the name is removed."""


def test_a_payload_the_child_cannot_rebuild_reports_why(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The send succeeds: every name in the payload resolves in this process.
    # The child is where one of them does not, which is the failure a
    # notebook-defined class produces and which no send-side check can catch.
    data = pickle.dumps(
        {
            "config": BackendConfig.model_validate({"method": "scipy/slsqp"}),
            "context": _make_context(),
            "initial_values": np.zeros(2),
            "marker": _VanishingMarker(),
        }
    )
    monkeypatch.delattr(sys.modules[__name__], "_VanishingMarker")

    ctx = multiprocessing.get_context("spawn")
    request_queue = ctx.Queue()
    result_queue = ctx.Queue()

    _run(data, request_queue, result_queue)

    payload = request_queue.get(timeout=5)
    # Queued, rather than the child dying before it reports anything: the
    # sentinel is what stops the parent waiting for a process that is gone.
    sentinel = request_queue.get(timeout=5)
    assert sentinel is None

    decoded = _decode_child_exception(payload)
    assert isinstance(decoded, AttributeError)
    assert any("Could not rebuild the optimization" in n for n in decoded.__notes__)


def test_the_advice_names_the_extra_only_when_it_is_missing() -> None:
    # Both directions have two forms, and the wrong one is worse than none:
    # advising an install the user already has explains nothing.
    for advice in (CANNOT_SERIALIZE, CANNOT_DESERIALIZE):
        assert ("ropt[cloudpickle]" in advice) is not HAVE_CLOUDPICKLE


def test_unserializable_problem_reports_how_to_send_it() -> None:
    class _Unserializable:
        # `_serialize` cuts the backend edge before dumping, so the stand-in
        # needs the field even though nothing reads it.
        backend = None

        def __reduce__(self) -> tuple[Any, ...]:
            msg = "cannot be sent"
            raise TypeError(msg)

    backend = ExternalBackend(
        BackendConfig.model_validate({"method": "external/scipy/slsqp"})
    )
    backend._context = cast("Any", _Unserializable())  # ruff: ignore[private-member-access]

    with pytest.raises(ExecutionError, match="could not be sent") as exc_info:
        backend._serialize(np.zeros(2))  # ruff: ignore[private-member-access]

    assert CANNOT_SERIALIZE in str(exc_info.value)


@pytest.mark.external
def test_a_closure_objective_runs_in_an_external_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The objective stays here, so it may be a closure whatever is installed.

    This is the one test that sends the graph a real run builds. The unit test
    above stops at an evaluator's bound method, while production inserts the
    `EnsembleOptimizer` and `EnsembleEvaluator` between the backend and that
    evaluator, and neither is covered there. Forcing the standard library here
    pins the whole graph: anything added along it that cannot be looked up by
    name fails this test rather than a user's run.
    """
    target = 0.5

    def _objective(variables: np.ndarray, _: Any) -> float:
        return float(((variables - target) ** 2).sum())

    config = {
        "optimizer": {"max_functions": 20},
        "backend": {
            "method": "external/slsqp",
            "max_iterations": 15,
            "convergence_tolerance": 1e-5,
        },
        "variables": {"variable_count": 3, "perturbation_magnitudes": 0.01},
    }
    dumped = _record_stdlib_dumps(monkeypatch, "ropt.backend.external.dumps")
    result = optimize(config, np.zeros(3), _objective)

    assert dumped
    assert result.variables is not None
    assert np.allclose(result.variables, target, atol=0.02)
