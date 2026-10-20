"""Tests for `run_program`, which runs an external program that an abort kills."""

# Every program here writes to a pipe that the test holds the read end of. A
# line on it proves the program is running, and end of file proves that every
# process that inherited it, the program and whatever it started, has exited.

from __future__ import annotations

import contextlib
import os
import select
import signal
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
import threading
from typing import TYPE_CHECKING

import numpy as np
import pytest

from ropt import AbortedError, run_program, session
from ropt.components.concurrency import AbortScope, _grace, _program
from ropt.enums import ExitCode

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from numpy.typing import NDArray

    from ropt import EvaluationFunctionContext

posix_only = pytest.mark.skipif(os.name != "posix", reason="needs POSIX")

# Starts a grandchild, reports both process IDs, and ignores SIGTERM, which the
# grandchild inherits, so that only SIGKILL ends either of them.
_STUBBORN = "trap '' TERM; sleep 60 & echo $$ $!; wait"


@pytest.fixture
def pipe() -> Iterator[tuple[int, int]]:
    read_fd, write_fd = os.pipe()
    yield read_fd, write_fd
    for fd in (read_fd, write_fd):
        with contextlib.suppress(OSError):
            os.close(fd)


@pytest.fixture
def short_grace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_grace, "GRACE_PERIOD", 0.2)


def _read_line(fd: int) -> str:
    data = b""
    while not data.endswith(b"\n"):
        readable, _, _ = select.select([fd], [], [], 30)
        assert readable, "the program did not report"
        chunk = os.read(fd, 1)
        assert chunk, "the program closed its output before reporting"
        data += chunk
    return data.decode()


def _assert_all_exited(read_fd: int, write_fd: int) -> None:
    os.close(write_fd)
    while True:
        readable, _, _ = select.select([read_fd], [], [], 30)
        assert readable, "a process holding the pipe is still running"
        if not os.read(read_fd, 1024):
            return


def _run_in_thread(
    scope: AbortScope, target: Callable[[], object]
) -> tuple[threading.Thread, list[BaseException]]:
    raised: list[BaseException] = []

    def run() -> None:
        with scope.as_current():
            try:
                target()
            except Exception as exc:  # ruff: ignore[blind-except]
                raised.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    return thread, raised


def test_run_program_returns_the_exit_status() -> None:
    assert run_program([sys.executable, "-c", "pass"]) == 0
    assert run_program([sys.executable, "-c", "raise SystemExit(3)"]) == 3


@posix_only
def test_run_program_reports_a_signal_as_a_negative_status() -> None:
    assert run_program(["sh", "-c", "kill -9 $$"]) == -signal.SIGKILL


def test_run_program_raises_for_a_missing_program(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_program([str(tmp_path / "missing")])


def test_run_program_rejects_a_single_string() -> None:
    with pytest.raises(TypeError, match="sequence"):
        run_program("true")


def test_run_program_rejects_a_pipe() -> None:
    with pytest.raises(ValueError, match="PIPE"):
        run_program([sys.executable, "-c", "pass"], stdout=subprocess.PIPE)


def test_run_program_writes_output_relative_to_cwd(tmp_path: Path) -> None:
    code = "import sys; print('out'); print('err', file=sys.stderr)"
    run_program(
        [sys.executable, "-c", code], cwd=tmp_path, stdout="out.txt", stderr="err.txt"
    )
    assert (tmp_path / "out.txt").read_text().strip() == "out"
    assert (tmp_path / "err.txt").read_text().strip() == "err"


def test_run_program_merges_stderr_into_stdout(tmp_path: Path) -> None:
    code = "import sys; print('out', flush=True); print('err', file=sys.stderr)"
    run_program(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        stdout="all.txt",
        stderr=subprocess.STDOUT,
    )
    assert (tmp_path / "all.txt").read_text().split() == ["out", "err"]


def test_run_program_passes_the_environment(pipe: tuple[int, int]) -> None:
    read_fd, write_fd = pipe
    code = "import os; print(os.environ['ROPT_PROBE'])"
    env = {**os.environ, "ROPT_PROBE": "value"}
    run_program([sys.executable, "-c", code], env=env, stdout=write_fd)
    assert _read_line(read_fd).strip() == "value"


@posix_only
@pytest.mark.timeout(60)
def test_run_program_abort_kills_the_program_and_its_children(
    pipe: tuple[int, int],
) -> None:
    read_fd, write_fd = pipe
    abort_scope = AbortScope()
    thread, raised = _run_in_thread(
        abort_scope,
        lambda: run_program(["sh", "-c", "sleep 60 & echo $!; wait"], stdout=write_fd),
    )
    _read_line(read_fd)
    abort_scope.abort(ExitCode.USER_ABORT)
    thread.join(timeout=30)

    assert not thread.is_alive()
    assert len(raised) == 1
    assert isinstance(raised[0], AbortedError)
    assert raised[0].exit_code is ExitCode.USER_ABORT
    _assert_all_exited(read_fd, write_fd)


@posix_only
@pytest.mark.timeout(60)
def test_run_program_abort_ends_the_grace_period_when_the_program_exits(
    pipe: tuple[int, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_grace, "GRACE_PERIOD", 45.0)
    read_fd, write_fd = pipe
    abort_scope = AbortScope()
    thread, raised = _run_in_thread(
        abort_scope,
        lambda: run_program(["sh", "-c", "echo ready; exec sleep 60"], stdout=write_fd),
    )
    _read_line(read_fd)
    abort_scope.abort()
    thread.join(timeout=15)

    assert not thread.is_alive()
    assert len(raised) == 1
    assert isinstance(raised[0], AbortedError)


@posix_only
@pytest.mark.slow
@pytest.mark.timeout(60)
@pytest.mark.usefixtures("short_grace")
def test_run_program_kills_a_program_that_ignores_sigterm(
    pipe: tuple[int, int],
) -> None:
    read_fd, write_fd = pipe
    abort_scope = AbortScope()
    thread, raised = _run_in_thread(
        abort_scope, lambda: run_program(["sh", "-c", _STUBBORN], stdout=write_fd)
    )
    _read_line(read_fd)
    abort_scope.abort()
    thread.join(timeout=30)

    assert not thread.is_alive()
    assert len(raised) == 1
    assert isinstance(raised[0], AbortedError)
    _assert_all_exited(read_fd, write_fd)


def test_run_program_does_not_start_once_aborted(tmp_path: Path) -> None:
    marker = tmp_path / "started"
    abort_scope = AbortScope()
    abort_scope.abort()
    with abort_scope.as_current(), pytest.raises(AbortedError):
        run_program([sys.executable, "-c", f"open({str(marker)!r}, 'w')"])
    assert not marker.exists()


@pytest.mark.slow
@posix_only
@pytest.mark.timeout(60)
def test_run_program_ends_with_the_process_that_started_it(
    pipe: tuple[int, int],
) -> None:
    # The wrapper notices that the socket to its starter closed, as it does when
    # a pool worker is killed.
    read_fd, write_fd = pipe
    code = (
        "import sys; from ropt import run_program; "
        "run_program(['sh', '-c', 'sleep 60 & echo $!; wait'])"
    )
    starter = subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]
        [sys.executable, "-c", code], stdout=write_fd
    )
    _read_line(read_fd)
    starter.kill()
    starter.wait()
    _assert_all_exited(read_fd, write_fd)


@posix_only
@pytest.mark.timeout(60)
def test_run_program_aborts_the_program_on_keyboard_interrupt(
    pipe: tuple[int, int],
) -> None:
    read_fd, write_fd = pipe

    def interrupt() -> None:
        _read_line(read_fd)
        signal.pthread_kill(threading.main_thread().ident or 0, signal.SIGINT)

    interrupter = threading.Thread(target=interrupt)
    interrupter.start()
    with pytest.raises(KeyboardInterrupt):
        run_program(["sh", "-c", "sleep 60 & echo $!; wait"], stdout=write_fd)
    interrupter.join(timeout=30)
    _assert_all_exited(read_fd, write_fd)


@pytest.mark.timeout(60)
@pytest.mark.usefixtures("short_grace")
def test_run_program_without_the_wrapper_terminates_the_program(
    monkeypatch: pytest.MonkeyPatch, pipe: tuple[int, int]
) -> None:
    monkeypatch.setattr(_program, "_use_wrapper", lambda: False)
    read_fd, write_fd = pipe
    assert run_program([sys.executable, "-c", "raise SystemExit(4)"]) == 4

    code = "import time; print('ready', flush=True); time.sleep(60)"
    abort_scope = AbortScope()
    thread, raised = _run_in_thread(
        abort_scope, lambda: run_program([sys.executable, "-c", code], stdout=write_fd)
    )
    _read_line(read_fd)
    abort_scope.abort()
    thread.join(timeout=30)

    assert not thread.is_alive()
    assert len(raised) == 1
    assert isinstance(raised[0], AbortedError)
    _assert_all_exited(read_fd, write_fd)


@posix_only
@pytest.mark.timeout(60)
def test_run_program_in_a_thread_pool_is_killed_by_session_abort(
    pipe: tuple[int, int],
) -> None:
    read_fd, write_fd = pipe
    config = {
        "variables": {"variable_count": 1, "perturbation_magnitudes": 0.01},
    }

    def objective(
        _variables: NDArray[np.float64], _context: EvaluationFunctionContext
    ) -> float:
        run_program(["sh", "-c", "sleep 60 & echo $!; wait"], stdout=write_fd)
        return 0.0

    with session() as opened:
        pool = opened.thread_pool(workers=1)

        def abort_when_running() -> None:
            _read_line(read_fd)
            opened.abort()

        aborter = threading.Thread(target=abort_when_running)
        aborter.start()
        with pytest.raises(AbortedError) as raised:
            pool.evaluate_batch(config, np.zeros((1, 1)), objective)
        aborter.join(timeout=30)

    assert raised.value.exit_code is ExitCode.USER_ABORT
    _assert_all_exited(read_fd, write_fd)
