"""Run one program and kill it, with everything it started, on request.

Started by `_program.Program` as `python -I <this file> <fd> <grace> -- <cmd>`,
never imported: it must start fast and stay out of reach of the caller's
packages, so it uses the standard library only.

The wrapper leads a process group of its own, and the program runs in it. A
socket connects it to the process that started it, and that socket closing,
by request or because that process died, is what aborts the program: the group
gets SIGTERM, and SIGKILL once the program has exited or `grace` seconds have
passed. The group ID is the wrapper's own PID, which cannot be reused while the
wrapper is alive to signal it.

Messages to the starter, one line each: `S` once the program has started,
`E <errno> <message>` if it could not be started, and `R <returncode>` once it
has exited unaborted.
"""

import contextlib
import os
import select
import signal
import socket
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
import time
from types import FrameType


def _send(sock: socket.socket, message: str) -> None:
    with contextlib.suppress(OSError):
        sock.sendall(f"{message}\n".encode())


def _exited_unaborted(
    process: subprocess.Popen[bytes], sock: socket.socket, wake_read: int
) -> bool:
    while True:
        readable, _, _ = select.select([sock.fileno(), wake_read], [], [])
        if sock.fileno() in readable:
            # Data or EOF: the starter sends nothing, so either means abort.
            return False
        data = os.read(wake_read, 64)
        if process.poll() is not None:
            break
        if b"t" in data:
            return False
    # Checked again, so that an abort racing a normal exit still kills
    # whatever the program left behind.
    readable, _, _ = select.select([sock.fileno()], [], [], 0)
    return not readable


def _abort(process: subprocess.Popen[bytes], wake_read: int, grace: float) -> None:
    os.killpg(0, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while process.poll() is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        readable, _, _ = select.select([wake_read], [], [], remaining)
        if readable:
            os.read(wake_read, 64)
    # Ends the wrapper as well, which is what tells the starter it is done.
    os.killpg(0, signal.SIGKILL)


def main() -> int:
    """Run the program named on the command line.

    Returns:
        The wrapper's exit status, which carries no meaning for the starter.
    """
    fd, grace, separator, *cmd = sys.argv[1:]
    if separator != "--" or not cmd:
        return 2
    sock = socket.socket(fileno=int(fd))
    wake_read, wake_write = os.pipe()

    def on_terminate(_signum: int, _frame: FrameType | None) -> None:
        # A SIGTERM aimed at the wrapper aborts like a closed socket; the one it
        # sends its own group when aborting arrives here too, and is harmless.
        os.write(wake_write, b"t")

    def on_child_exit(_signum: int, _frame: FrameType | None) -> None:
        os.write(wake_write, b"x")

    signal.signal(signal.SIGTERM, on_terminate)
    # Installed before the program starts, so that an early exit is not missed.
    signal.signal(signal.SIGCHLD, on_child_exit)
    try:
        process = subprocess.Popen(cmd)  # ruff: ignore[subprocess-without-shell-equals-true]
    except OSError as exc:
        _send(sock, f"E {exc.errno or 0} {exc.strerror or exc}")
        return 1
    _send(sock, "S")
    if _exited_unaborted(process, sock, wake_read):
        _send(sock, f"R {process.returncode}")
        return 0
    _abort(process, wake_read, float(grace))
    return 1


if __name__ == "__main__":
    sys.exit(main())
