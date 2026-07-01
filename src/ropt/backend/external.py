"""This module implements an optimization plugin that employs an external processes."""

from __future__ import annotations

import multiprocessing
import queue
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
_QUEUE_POLL_INTERVAL: Final = 1.0


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
        self._is_parallel: bool = backend.is_parallel

    def start(  # noqa: C901, D102
        self, initial_values: NDArray[np.float64]
    ) -> None:
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
            try:
                request = request_queue.get(timeout=_QUEUE_POLL_INTERVAL)
            except queue.Empty:
                if not process.is_alive():
                    exception = RuntimeError(
                        "External backend subprocess died unexpectedly "
                        f"(exit code {process.exitcode})"
                    )
                    break
                continue
            if request is None:
                break
            outcome = _handle_request(request)
            if isinstance(outcome, Exception):
                exception = outcome
                break
            try:
                result = self._optimizer_callback(
                    outcome["variables"],
                    return_functions=outcome["return_functions"],
                    return_gradients=outcome["return_gradients"],
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
    except Abort as exc:
        request_queue.put({"abort": True, "exit_code": int(exc.exit_code)})
    except Exception as exc:  # noqa: BLE001
        request_queue.put(_encode_child_exception(exc))
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


def _handle_request(
    request: dict[str, Any],
) -> Exception | dict[str, Any]:
    if "abort" in request:
        return Abort(exit_code=ExitCode(request["exit_code"]))
    if "exception" in request:
        return _decode_child_exception(request)
    if "error" in request:
        return _wrap_with_traceback(
            f"External backend subprocess raised {request['error']}: {request['message']}",
            request["traceback"],
        )
    return request


def _encode_child_exception(exc: BaseException) -> dict[str, Any]:
    tb_str = traceback.format_exc()
    try:
        pickled = cloudpickle.dumps(exc)
    except Exception:  # noqa: BLE001
        return {
            "error": type(exc).__name__,
            "message": str(exc),
            "traceback": tb_str,
        }
    return {"exception": pickled, "traceback": tb_str}


def _decode_child_exception(request: dict[str, Any]) -> Exception:
    tb_str = request.get("traceback", "")
    try:
        original = cloudpickle.loads(request["exception"])
    except Exception:  # noqa: BLE001
        return _wrap_with_traceback(
            "External backend exception could not be deserialized", tb_str
        )

    if not isinstance(original, Exception):
        return _wrap_with_traceback(
            f"External backend subprocess raised {type(original).__name__}: {original!r}",
            tb_str,
        )

    if tb_str:
        original.add_note(f"External backend child traceback:\n{tb_str}")
    return original


def _wrap_with_traceback(message: str, tb_str: str) -> RuntimeError:
    err = RuntimeError(message)
    if tb_str:
        err.add_note(f"Child traceback:\n{tb_str}")
    return err
