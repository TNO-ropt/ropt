"""This module implements an optimization plugin that employs an external processes."""

from __future__ import annotations

import atexit
import contextlib
import json
import os
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Final, Self

import numpy as np

from ropt.config.enopt import EnOptConfig
from ropt.exceptions import ConfigError, OptimizationAborted
from ropt.plugins import PluginManager

from .base import Optimizer, OptimizerCallback, OptimizerPlugin

if TYPE_CHECKING:
    from numpy.typing import NDArray


_PLUGIN_BINARY: Final = "ropt_plugin_optimizer"
_PROCESS_TIMEOUT: Final = 10


class ExternalOptimizer(Optimizer):
    """Plugin class for optimization using an external process.

    This class enables optimization via an external process, which performs the
    optimization independently and communicates with this class over pipes to
    request function evaluations, report optimizer states, and handle any
    errors.

    Typically, the optimizer is specified within an
    [`OptimizerConfig`][ropt.config.enopt.OptimizerConfig] via the `method`
    field, which either provides the algorithm name directly or follows the form
    `plugin-name/method-name`. In the first case, `ropt` searches among all
    available optimizer plugins to find the specified method. In the second
    case, it checks if the plugin identified by `plugin-name` contains
    `method-name` and, if so, uses it. Both of these are not supported by the
    external optimizer class. Instead, it requires that the `method` field
    includes both the plugin and method names in the format
    `external/plugin-name/method-name` or `external/method-name`. This ensures
    the external optimizer can identify and launch the specified optimization
    method `method-name` and launch it as an external process.
    """

    def __init__(
        self, config: EnOptConfig, optimizer_callback: OptimizerCallback
    ) -> None:
        """Initialize the optimizer.

        See the [ropt.plugins.optimizer.base.Optimizer][] abstract base class.

        # noqa
        """
        self._config = config
        self._optimizer_callback = optimizer_callback
        self._process_pid: int | None = None

        optimizer: Optimizer = (
            PluginManager()
            .get_plugin("optimizer", config.optimizer.method.split("/", maxsplit=1)[1])
            .create(config, lambda *_: None)
        )
        self._allow_nan = optimizer.allow_nan
        self._is_parallel = optimizer.is_parallel
        del optimizer

    def start(self, initial_values: NDArray[np.float64]) -> None:
        """Start the optimization.

        See the [ropt.plugins.optimizer.base.Optimizer][] abstract base class.

        # noqa
        """
        atexit.register(self._atexit)

        with TemporaryDirectory() as fifo_dir:
            fifo1 = Path(fifo_dir) / "fifo1"
            fifo2 = Path(fifo_dir) / "fifo2"

            with _JSONPipeCommunicator(fifo1, fifo2) as comm:
                try:
                    process = subprocess.Popen(  # noqa: S603
                        [_PLUGIN_BINARY, str(fifo2), str(fifo1), str(os.getpid())]
                    )
                    self._process_pid = process.pid
                except FileNotFoundError as exc:
                    msg = "The plugin runner binary was not found"
                    raise ConfigError(msg) from exc

                answer: str | list[Any] | dict[str, Any] | None = None
                exception: BaseException | None = None

                while process.poll() is None:
                    if answer is None:
                        try:
                            answer = self._handle_request(comm, initial_values)
                        except Exception as exc:  # noqa: BLE001
                            # Store the exception, we first need to send the 'abort' signal:
                            exception = exc
                            answer = "abort"

                    if answer is not None and comm.write(answer):
                        answer = None
                        # If the message has been sent, then reraise any exceptions:
                        if exception is not None:
                            # The process should have aborted:
                            with contextlib.suppress(ProcessLookupError):
                                os.kill(self._process_pid, signal.SIGTERM)
                            with contextlib.suppress(subprocess.TimeoutExpired):
                                process.wait(_PROCESS_TIMEOUT)
                            raise exception
                    time.sleep(0.1)

                with contextlib.suppress(ProcessLookupError):
                    os.kill(self._process_pid, signal.SIGTERM)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(_PROCESS_TIMEOUT)

    @property
    def allow_nan(self) -> bool:
        """Whether NaN is allowed.

        See the [ropt.plugins.optimizer.base.Optimizer][] abstract base class.

        # noqa
        """
        return self._allow_nan

    @property
    def is_parallel(self) -> bool:
        """Whether the current run is parallel.

        See the [ropt.plugins.optimizer.base.Optimizer][] abstract base class.

        # noqa
        """
        return self._is_parallel

    def _handle_request(
        self, comm: _JSONPipeCommunicator, initial_values: NDArray[np.float64]
    ) -> str | list[Any] | dict[str, Any] | None:
        request = comm.read()

        if request == "config":
            return self._config.model_dump(round_trip=True)

        if request == "initial_values":
            return initial_values.tolist()  # type: ignore[no-any-return]

        if isinstance(request, dict):
            if (evaluation := request.get("evaluation")) is not None:
                functions, gradients = self._optimizer_callback(
                    np.array(evaluation["variables"], dtype=np.float64),
                    return_functions=evaluation["return_functions"],
                    return_gradients=evaluation["return_gradients"],
                )
                return {
                    "functions": functions.tolist(),
                    "gradients": gradients.tolist(),
                }
            if (error := request.get("error")) is not None:
                msg = f"External optimizer error: {error}"
                raise RuntimeError(msg)

        return None

    def _atexit(self) -> None:
        if self._process_pid is not None:
            with contextlib.suppress(ProcessLookupError):
                os.kill(self._process_pid, signal.SIGTERM)


class ExternalOptimizerPlugin(OptimizerPlugin):
    """The external optimizer plugin class."""

    @classmethod
    def create(
        cls, config: EnOptConfig, optimizer_callback: OptimizerCallback
    ) -> ExternalOptimizer:
        """Initialize the optimizer plugin.

        See the [ropt.plugins.optimizer.base.OptimizerPlugin][] abstract base class.

        # noqa
        """
        return ExternalOptimizer(config, optimizer_callback)

    @classmethod
    def is_supported(cls, method: str) -> bool:
        """Check if a method is supported.

        See the [ropt.plugins.base.Plugin][] abstract base class.

        # noqa
        """
        return PluginManager().is_supported("optimizer", method)

    @classmethod
    def allows_discovery(cls) -> bool:
        """Check if the plugin can be discovered automatically.

        See the [ropt.plugins.optimizer.base.OptimizerPlugin][] abstract base class.

        # noqa
        """
        return False

    @classmethod
    def validate_options(
        cls, method: str, options: dict[str, Any] | list[str] | None
    ) -> None:
        """Validate the options of a given method.

        See the [ropt.plugins.optimizer.base.OptimizerPlugin][] abstract base class.

        # noqa
        """
        method = method.split("/", maxsplit=1)[1]
        PluginManager().get_plugin("optimizer", method).validate_options(
            method, options
        )


class _PluginOptimizer:
    def __init__(self, parent_pid: int) -> None:
        self._parent_pid = parent_pid
        self._comm: _JSONPipeCommunicator

    def _check_parent(self) -> None:
        try:
            os.kill(self._parent_pid, 0)
        except OSError:
            sys.exit(0)

    def _request(self, request: Any) -> Any:  # noqa: ANN401
        while True:
            self._check_parent()
            if self._comm.write(request):
                break
        while True:
            self._check_parent()
            answer = self._comm.read()
            if answer is not None:
                return answer

    def _callback(
        self,
        variables: NDArray[np.float64],
        *,
        return_functions: bool,
        return_gradients: bool,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        response = self._request(
            {
                "evaluation": {
                    "variables": variables.tolist(),
                    "return_functions": return_functions,
                    "return_gradients": return_gradients,
                }
            },
        )
        if response == "abort":
            raise OptimizationAborted(response)
        return (
            np.array(response["functions"], dtype=np.float64),
            np.array(response["gradients"], dtype=np.float64),
        )

    def run(self, fifo1: Path, fifo2: Path) -> int:
        with _JSONPipeCommunicator(fifo1, fifo2) as self._comm:
            config = EnOptConfig.model_validate(self._request("config"))

            optimizer = (
                PluginManager()
                .get_plugin(
                    "optimizer", config.optimizer.method.split("/", maxsplit=1)[1]
                )
                .create(config, self._callback)
            )
            try:
                optimizer.start(
                    np.array(self._request("initial_values"), dtype=np.float64)
                )
            except OptimizationAborted:
                return 0
            except Exception as exc:  # noqa: BLE001
                assert self._request({"error": str(exc)}) == "abort"  # noqa: PT017
                return 1
        return 0


def ropt_plugin_optimizer() -> int:
    """Entry point for the plugin runner script.

    Returns:
        The exit code of the script.
    """
    fifo1 = Path(sys.argv[1])
    fifo2 = Path(sys.argv[2])
    parent_pid = int(sys.argv[3])

    assert fifo1.exists()
    assert fifo2.exists()

    return _PluginOptimizer(parent_pid).run(fifo1, fifo2)


class _JSONPipeCommunicator:
    DELIMITER = "--READY--"

    def __init__(self, read_pipe: Path, write_pipe: Path, timeout: float = 1.0) -> None:
        self._read_pipe = read_pipe
        self._write_pipe = write_pipe
        self._timeout = timeout

        self._read_fd: int
        self._write_fd: int | None
        self._selector: selectors.BaseSelector

        if not read_pipe.exists():
            os.mkfifo(read_pipe)
        if not write_pipe.exists():
            os.mkfifo(write_pipe)

    def __enter__(self) -> Self:
        self._read_fd = os.open(self._read_pipe, os.O_RDONLY | os.O_NONBLOCK)
        self._write_fd = None
        self._selector = selectors.DefaultSelector()
        self._selector.register(self._read_fd, selectors.EVENT_READ)
        return self

    def __exit__(self, *args: object) -> None:
        self._selector.close()
        os.close(self._read_fd)
        if self._write_fd is not None:
            os.close(self._write_fd)

    def read(self) -> str | list[Any] | dict[str, Any] | None:
        events = self._selector.select(timeout=self._timeout)
        for _, mask in events:
            if mask & selectors.EVENT_READ:
                with os.fdopen(os.dup(self._read_fd), "r", encoding="utf-8") as fd:
                    buffer = ""
                    while line := fd.readline():
                        if line.strip() == self.DELIMITER:
                            buffer = buffer.strip()
                            return json.loads(buffer) if buffer else buffer
                        buffer += line
        return None

    def write(self, data: str | list[Any] | dict[str, Any]) -> bool:
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj: Any) -> Any:  # noqa: ANN401
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, Path):
                    return str(obj)
                return super().default(obj)

        if self._write_fd is None:
            self._write_fd = os.open(self._write_pipe, os.O_WRONLY | os.O_NONBLOCK)
            self._selector.register(self._write_fd, selectors.EVENT_WRITE)
        events = self._selector.select(timeout=self._timeout)
        for _, mask in events:
            if mask & selectors.EVENT_WRITE:
                os.write(
                    self._write_fd,
                    f"{json.dumps(data, cls=CustomEncoder)}\n{self.DELIMITER}\n".encode(),
                )
                return True
        return False
