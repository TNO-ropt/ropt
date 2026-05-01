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
    """Backend implementation that runs an optimizer in a separate process.

    Implements the [`Backend`][ropt.backend.Backend] interface by spawning a
    child process to run a delegate backend. The child process performs the
    optimization independently and communicates back through queues to request
    function evaluations, report optimizer states, and propagate errors.

    **Method naming**

    Unlike other backends, the `method` field of
    [`BackendConfig`][ropt.config.BackendConfig] must include both the plugin
    and method name in one of these forms:

    - `external/plugin-name/method-name`
    - `external/method-name`

    The `external/` prefix is stripped before the remainder is forwarded to
    the delegate plugin. Standard `plugin-name/method-name` resolution without
    the prefix is not supported by this backend.

    Note:
        The `cloudpickle` package must be installed. If it is absent,
        instantiation raises `NotImplementedError`.
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
