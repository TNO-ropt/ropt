"""This module implements an optimization plugin that employs an external processes."""

from __future__ import annotations

import multiprocessing
import traceback
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from ropt.core import OptimizerCallback, OptimizerCallbackResult
from ropt.enums import ExitCode
from ropt.exceptions import ComputeStepAborted
from ropt.optimizer import Optimizer
from ropt.plugins.manager import get_plugin

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ropt.config import EnOptConfig


_PROCESS_TIMEOUT: Final = 10


class ExternalOptimizer(Optimizer):
    """Plugin class for optimization using an external process.

    This class enables optimization via an external process, which performs the
    optimization independently and communicates with this class to request
    function evaluations, report optimizer states, and handle any errors.

    Typically, the optimizer is specified within an
    [`OptimizerConfig`][ropt.config.OptimizerConfig] via the `method` field,
    which either provides the algorithm name directly or follows the form
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

        See the [ropt.optimizer.Optimizer][] abstract base class.

        # noqa
        """
        self._config = config
        self._optimizer_callback = optimizer_callback
        self._process_pid: int | None = None

        optimizer: Optimizer = get_plugin(
            "optimizer", config.optimizer.method.split("/", maxsplit=1)[1]
        ).create(config, lambda *_: None)
        self._allow_nan = optimizer.allow_nan
        self._is_parallel = optimizer.is_parallel
        del optimizer

    def start(self, initial_values: NDArray[np.float64]) -> None:
        """Start the optimization.

        See the [ropt.optimizer.Optimizer][] abstract base class.

        # noqa
        """
        context = multiprocessing.get_context("spawn")
        request_queue = context.Queue()
        result_queue = context.Queue()
        process = context.Process(
            target=_run,
            args=(self._config, initial_values, request_queue, result_queue),
        )

        result: OptimizerCallbackResult | ExitCode
        exception: Exception | None = None

        process.start()
        while exception is None:
            if (request := request_queue.get()) is None:
                break
            if "error" in request:
                error_type = request["error"]
                message = request["message"]
                tb_str = request["traceback"]
                exception = RuntimeError(
                    f"External optimizer error: {error_type}\nmessage:{message}\nstacktrace: {tb_str}"
                )
                break
            try:
                result = self._optimizer_callback(
                    request["variables"],
                    return_functions=request["return_functions"],
                    return_gradients=request["return_gradients"],
                )
            except Exception as exc:  # noqa: BLE001
                result = ExitCode.ABORT_FROM_ERROR
                exception = exc
            result_queue.put(result)
        process.join(_PROCESS_TIMEOUT)

        if process.is_alive():
            process.terminate()
            process.join(_PROCESS_TIMEOUT)
            if process.is_alive():
                try:
                    process.kill()
                    process.join(_PROCESS_TIMEOUT)
                except AttributeError:
                    pass

        if exception is not None:
            raise exception

    @property
    def allow_nan(self) -> bool:
        """Whether NaN is allowed.

        See the [ropt.optimizer.Optimizer][] abstract base class.

        # noqa
        """
        return self._allow_nan

    @property
    def is_parallel(self) -> bool:
        """Whether the current run is parallel.

        See the [ropt.optimizer.Optimizer][] abstract base class.

        # noqa
        """
        return self._is_parallel


def _run(
    config: EnOptConfig,
    initial_values: NDArray[np.float64],
    request_queue: multiprocessing.Queue[dict[str, Any] | None],
    result_queue: multiprocessing.Queue[OptimizerCallbackResult | ExitCode],
) -> None:
    optimizer = _PluginOptimizer(config, initial_values, request_queue, result_queue)
    optimizer.run()


class _PluginOptimizer:
    def __init__(
        self,
        config: EnOptConfig,
        initial_values: NDArray[np.float64],
        request_queue: multiprocessing.Queue[dict[str, Any] | None],
        result_queue: multiprocessing.Queue[OptimizerCallbackResult | ExitCode],
    ) -> None:
        self._config = config
        self._initial_values = np.asarray(initial_values, dtype=np.float64)
        self._request_queue = request_queue
        self._result_queue = result_queue

    def _callback(
        self,
        variables: NDArray[np.float64],
        *,
        return_functions: bool,
        return_gradients: bool,
    ) -> OptimizerCallbackResult:
        self._request_queue.put(
            {
                "variables": variables,
                "return_functions": return_functions,
                "return_gradients": return_gradients,
            },
        )
        result = self._result_queue.get()
        if isinstance(result, OptimizerCallbackResult):
            return result
        assert isinstance(result, ExitCode)
        raise ComputeStepAborted(exit_code=result)

    def run(self) -> None:
        optimizer = get_plugin(
            "optimizer", self._config.optimizer.method.split("/", maxsplit=1)[1]
        ).create(self._config, self._callback)
        try:
            optimizer.start(self._initial_values)
        except ComputeStepAborted:
            pass
        except Exception as exc:  # noqa: BLE001
            tb_str = traceback.format_exc()
            self._request_queue.put(
                {"error": type(exc).__name__, "message": str(exc), "traceback": tb_str}
            )
        finally:
            self._request_queue.put(None)
