"""Start external programs that an abort can kill, with whatever they started."""

from __future__ import annotations

import contextlib
import os
import select
import socket
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
import threading
from contextlib import ExitStack
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from ropt.exceptions import AbortedError, ExecutionError

from . import _grace
from ._abort_scope import current_abort_scope

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

type ProgramOutput = str | os.PathLike[str] | int | IO[Any] | None

_WRAPPER = Path(__file__).with_name("_program_wrapper.py")

# How long a wait may block before Python gets to run a pending signal handler,
# which some extension modules otherwise hold off until the wait ends.
_POLL_INTERVAL = 0.1


def _use_wrapper() -> bool:
    return os.name == "posix" and bool(sys.executable)


class Program:
    """A running external program that can be aborted.

    On POSIX the program runs under a small wrapper process in a process group
    of its own. Aborting it, or the death of this process, sends SIGTERM to that
    group, and SIGKILL once the program has exited or the grace period has
    passed. Elsewhere only the program itself is terminated and then killed, and
    whatever it started may outlive it.
    """

    def __init__(
        self,
        cmd: Sequence[str],
        *,
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        stdout: int | IO[Any] | None = None,
        stderr: int | IO[Any] | None = None,
    ) -> None:
        """Start the program.

        Args:
            cmd:    The program and its arguments.
            cwd:    The working directory, or `None` to inherit it.
            env:    The environment, or `None` to inherit it.
            stdout: Where standard output goes, as for `subprocess.Popen`.
            stderr: Where standard error goes, as for `subprocess.Popen`.
        """
        self._lock = threading.Lock()
        self._aborted = False
        self._returncode: int | None = None
        self._grace = _grace.GRACE_PERIOD
        self._sock: socket.socket | None = None
        self._buffer = b""
        if _use_wrapper():
            self._process = self._start_wrapper(cmd, cwd, env, stdout, stderr)
        else:
            self._process = subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]
                list(cmd),
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
            )

    def _start_wrapper(
        self,
        cmd: Sequence[str],
        cwd: str | os.PathLike[str] | None,
        env: Mapping[str, str] | None,
        stdout: int | IO[Any] | None,
        stderr: int | IO[Any] | None,
    ) -> subprocess.Popen[bytes]:
        ours, theirs = socket.socketpair()
        try:
            process = subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]
                [
                    sys.executable,
                    "-I",
                    str(_WRAPPER),
                    str(theirs.fileno()),
                    repr(self._grace),
                    "--",
                    *cmd,
                ],
                pass_fds=(theirs.fileno(),),
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
        except BaseException:
            ours.close()
            raise
        finally:
            theirs.close()
        self._sock = ours
        try:
            message = self._read_message(blocking=True)
        except BaseException:
            self.abort()
            raise
        if message is not None and message.startswith("S"):
            return process
        process.wait()
        ours.close()
        if message is not None and message.startswith("E "):
            _, errno, strerror = message.split(" ", 2)
            raise OSError(int(errno), strerror, cmd[0])
        msg = f"The program wrapper failed to start: {cmd[0]}"
        raise ExecutionError(msg)

    def _read_message(self, *, blocking: bool) -> str | None:
        # Returns one line, or None at EOF or, when not blocking, if none is
        # complete yet.
        assert self._sock is not None
        while b"\n" not in self._buffer:
            if not blocking:
                readable, _, _ = select.select([self._sock], [], [], 0)
                if not readable:
                    return None
            data = self._sock.recv(256)
            if not data:
                return None
            self._buffer += data
        line, _, self._buffer = self._buffer.partition(b"\n")
        return line.decode()

    @property
    def pid(self) -> int:
        """The process ID of the wrapper, or of the program without one.

        Returns:
            The process ID.
        """
        return self._process.pid

    @property
    def aborted(self) -> bool:
        """Whether `abort` was called.

        Returns:
            `True` once the program was aborted.
        """
        with self._lock:
            return self._aborted

    def abort(self) -> None:
        """Abort the program, without waiting for it to end.

        Safe to call from any thread, more than once, and after the program has
        ended.
        """
        with self._lock:
            if self._aborted:
                return
            self._aborted = True
        if self._sock is not None:
            with contextlib.suppress(OSError):
                self._sock.shutdown(socket.SHUT_WR)
            return
        if self._process.poll() is None:
            with contextlib.suppress(OSError):
                self._process.terminate()
            threading.Thread(target=self._kill_after_grace, daemon=True).start()

    def _kill_after_grace(self) -> None:
        try:
            self._process.wait(timeout=self._grace)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError):
                self._process.kill()

    def poll(self) -> bool:
        """Return whether the program has ended, without waiting.

        Returns:
            `True` once the program, and on POSIX its wrapper, have ended.
        """
        if self._process.poll() is None:
            return False
        self._finish()
        return True

    def wait(self) -> None:
        """Wait for the program to end.

        The wait is broken into short ones, so that a pending Ctrl-C is raised
        while it lasts.
        """
        while True:
            if self._sock is not None:
                readable, _, _ = select.select([self._sock], [], [], _POLL_INTERVAL)
                if readable and not self._sock.recv(256, socket.MSG_PEEK):
                    self._process.wait()
                    break
                if readable:
                    self._buffer += self._sock.recv(256)
                continue
            try:
                self._process.wait(timeout=_POLL_INTERVAL)
                break
            except subprocess.TimeoutExpired:
                continue
        self._finish()

    def _finish(self) -> None:
        if self._sock is None:
            self._returncode = self._process.returncode
            return
        with contextlib.suppress(OSError):
            message = self._read_message(blocking=False)
            if message is not None and message.startswith("R "):
                self._returncode = int(message[2:])
        self._sock.close()

    @property
    def returncode(self) -> int | None:
        """The exit status of a program that ended without being aborted.

        Returns:
            The exit status, negative for a signal as in `subprocess`, or `None`.
        """
        return self._returncode

    def detach(self) -> None:
        """Stop tracking an aborted program, and reap it in the background."""
        if self._sock is not None:
            self._sock.close()
        threading.Thread(target=self._process.wait, daemon=True).start()


def run_program(
    cmd: Sequence[str | os.PathLike[str]],
    *,
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    stdout: ProgramOutput = None,
    stderr: ProgramOutput = None,
) -> int:
    """Run an external program, and stop it when its caller is stopped.

    Blocks until the program exits. The program, and everything it started, is
    terminated when the task that runs the caller is aborted, when an exception
    such as `KeyboardInterrupt` interrupts the call, or when the calling process
    ends. An abort of the task raises
    [`AbortedError`][ropt.exceptions.AbortedError]. Code that no task runs is
    not reached by an abort.

    `stdout` and `stderr` take `None` to inherit the caller's, a path to write
    to, an open file, a file descriptor, or `subprocess.DEVNULL`; `stderr` also
    takes `subprocess.STDOUT`. A relative path is relative to `cwd`. Standard
    input is empty. See [Running External Programs](../running/programs.md).

    Args:
        cmd:    The program and its arguments.
        cwd:    The working directory, or `None` to inherit it.
        env:    The environment, or `None` to inherit it.
        stdout: Where standard output goes.
        stderr: Where standard error goes.

    Returns:
        The exit status of the program, negative for a signal as in `subprocess`.

    Raises:
        AbortedError: If the calling task was aborted.
        OSError:      If the program could not be started.
        TypeError:    If `cmd` is a single string.
        ValueError:   If `stdout` or `stderr` is `subprocess.PIPE`.
    """  # ruff: ignore[docstring-extraneous-exception]
    if isinstance(cmd, (str, bytes)):
        msg = "cmd must be a sequence of arguments, not a single string"
        raise TypeError(msg)
    args = [os.fspath(arg) for arg in cmd]
    scope = current_abort_scope()
    if scope is not None and scope.aborting:
        raise AbortedError(scope.exit_code)
    with ExitStack() as stack:
        out = _output(stdout, cwd, stack, name="stdout")
        err = _output(stderr, cwd, stack, name="stderr")
        program = Program(args, cwd=cwd, env=env, stdout=out, stderr=err)
    if scope is not None:
        scope.add_callback(program.abort)
    try:
        program.wait()
    except BaseException:
        program.abort()
        program.detach()
        raise
    finally:
        if scope is not None:
            scope.remove_callback(program.abort)
    if program.aborted:
        assert scope is not None
        raise AbortedError(scope.exit_code)
    assert program.returncode is not None
    return program.returncode


def _output(
    target: ProgramOutput,
    cwd: str | os.PathLike[str] | None,
    stack: ExitStack,
    *,
    name: str,
) -> int | IO[Any] | None:
    if target == subprocess.PIPE:
        msg = f"{name} cannot be subprocess.PIPE"
        raise ValueError(msg)
    if target == subprocess.STDOUT and name == "stdout":
        msg = "stdout cannot be subprocess.STDOUT"
        raise ValueError(msg)
    if not isinstance(target, (str, os.PathLike)):
        return target
    path = Path(target)
    if cwd is not None and not path.is_absolute():
        path = Path(cwd) / path
    return stack.enter_context(path.open("wb"))
