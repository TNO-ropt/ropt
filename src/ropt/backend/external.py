"""This module implements an optimization plugin that employs an external processes."""

from __future__ import annotations

import multiprocessing
import traceback
from functools import partial
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from ropt.backend import Backend
from ropt.core import OptimizerCallback, OptimizerCallbackResult
from ropt.enums import ExitCode
from ropt.exceptions import Abort
from ropt.plugins.manager import get_plugin

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ropt.config import BackendConfig
    from ropt.context import EnOptContext

_HAVE_CLOUDPICKLE: Final = find_spec("cloudpickle") is not None

if _HAVE_CLOUDPICKLE:
    import cloudpickle


_PROCESS_TIMEOUT: Final = 10


class ExternalBackend(Backend):
    """Plugin class for optimization using an external process.

    This class enables optimization via an external process, which performs the
    optimization independently and communicates with this class to request
    function evaluations, report optimizer states, and handle any errors.

    Typically, the optimizer is specified within an
    [`BackendConfig`][ropt.config.BackendConfig] via the `method` field,
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

    def __init__(self, backend_config: BackendConfig) -> None:  # noqa: D107
        if not _HAVE_CLOUDPICKLE:
            msg = "The cloudpickle module must be installed to use ExternalBackend"
            raise NotImplementedError(msg)

        self._backend_config = backend_config.model_copy(
            update={"method": backend_config.method.split("/", maxsplit=1)[1]}
        )
        self._backend_plugin = get_plugin("backend", method=self._backend_config.method)

    def init(  # noqa: D102
        self, context: EnOptContext, optimizer_callback: OptimizerCallback
    ) -> None:
        self._context = context
        self._optimizer_callback = optimizer_callback
        backend = self._backend_plugin.create(self._backend_config)
        backend.init(
            context.model_copy(update={"backend": backend}), optimizer_callback
        )
        self._allow_nan: bool = backend.allow_nan
        self._is_parallel: bool = backend.is_parallel

    def start(self, initial_values: NDArray[np.float64]) -> None:  # noqa: D102
        context = multiprocessing.get_context("spawn")
        request_queue = context.Queue()
        result_queue = context.Queue()

        process = context.Process(
            target=_run,
            args=(
                cloudpickle.dumps(
                    {
                        "config": self._backend_config,
                        "context": self._context,
                        "initial_values": initial_values,
                    }
                ),
                request_queue,
                result_queue,
            ),
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

    def validate_options(  # noqa: D102
        self,
    ) -> None:
        self._backend_plugin.create(self._backend_config).validate_options()

    @property
    def allow_nan(self) -> bool:  # noqa: D102
        return self._allow_nan

    @property
    def is_parallel(self) -> bool:  # noqa: D102
        return self._is_parallel


def _run(
    data: bytes,
    request_queue: multiprocessing.Queue[dict[str, Any] | None],
    result_queue: multiprocessing.Queue[OptimizerCallbackResult | ExitCode],
) -> None:
    data_dict = cloudpickle.loads(data)
    config = data_dict["config"]
    context = data_dict["context"]
    initial_values = data_dict["initial_values"]

    backend_plugin = get_plugin("backend", method=config.method)
    backend = backend_plugin.create(config)
    context = context.model_copy(update={"backend": backend})
    backend.init(
        context,
        partial(_callback, request_queue=request_queue, result_queue=result_queue),
    )

    try:
        backend.start(np.asarray(initial_values, dtype=np.float64))
    except Abort:
        pass
    except Exception as exc:  # noqa: BLE001
        tb_str = traceback.format_exc()
        request_queue.put(
            {"error": type(exc).__name__, "message": str(exc), "traceback": tb_str}
        )
    finally:
        request_queue.put(None)


def _callback(
    variables: NDArray[np.float64],
    *,
    return_functions: bool,
    return_gradients: bool,
    request_queue: multiprocessing.Queue[dict[str, Any] | None],
    result_queue: multiprocessing.Queue[OptimizerCallbackResult | ExitCode],
) -> OptimizerCallbackResult:
    request_queue.put(
        {
            "variables": variables,
            "return_functions": return_functions,
            "return_gradients": return_gradients,
        },
    )
    result = result_queue.get()
    if isinstance(result, OptimizerCallbackResult):
        return result
    assert isinstance(result, ExitCode)
    raise Abort(exit_code=result)
