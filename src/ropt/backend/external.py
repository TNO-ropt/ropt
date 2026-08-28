"""This module implements an optimization plugin that employs an external processes."""

from __future__ import annotations

import contextlib
import multiprocessing
import queue
import traceback
from functools import partial
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from ropt._logging import get_logger
from ropt._serialize import CANNOT_DESERIALIZE, CANNOT_SERIALIZE, dumps, loads
from ropt.backend import Backend
from ropt.exceptions import ExecutionError, OptimizerStop
from ropt.plugins.manager import get_plugin

if TYPE_CHECKING:
    from multiprocessing.process import BaseProcess

    from numpy.typing import NDArray

    from ropt.config import BackendConfig
    from ropt.context import EnOptContext
    from ropt.core import OptimizerCallback, OptimizerCallbackResult

_logger = get_logger(__name__)

_PROCESS_TIMEOUT: Final = 10
_QUEUE_POLL_INTERVAL: Final = 1.0


class _DriverStopped(Exception):  # ruff: ignore[error-suffix-on-exception-name]
    """Unwinds the child optimizer after a driver-side evaluation failure.

    Swallowed in the child and never reported back; the driver re-raises the
    real exception.
    """


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
        The problem is sent to the child process by serializing it, so
        everything the delegate needs must be serializable. The standard
        library can send anything that can be looked up by name, which covers
        the built-in plugins and any plugin class defined in an importable
        module. Installing the optional `cloudpickle` extra lifts that
        restriction, so plugin instances of classes defined inside a function
        or a notebook can be sent as well.
    """

    def __init__(self, backend_config: BackendConfig) -> None:
        """Initialize the external backend.

        Args:
            backend_config: The backend configuration; its `method` field must
                            be prefixed with `external/`.
        """
        self._backend_config = backend_config.model_copy(
            update={"method": backend_config.method.split("/", maxsplit=1)[1]}
        )
        self._backend_plugin = get_plugin("backend", method=self._backend_config.method)

    def init(  # ruff: ignore[undocumented-public-method]
        self, context: EnOptContext, optimizer_callback: OptimizerCallback
    ) -> None:
        self._context = context
        self._optimizer_callback = optimizer_callback
        backend = self._backend_plugin.create(self._backend_config)
        backend.init(
            context.model_copy(update={"backend": backend}), optimizer_callback
        )
        self._is_parallel: bool = backend.is_parallel

    def start(  # ruff: ignore[undocumented-public-method]
        self, initial_values: NDArray[np.float64]
    ) -> None:
        payload = self._serialize(initial_values)

        context = multiprocessing.get_context("spawn")
        request_queue = context.Queue()
        result_queue = context.Queue()

        process = context.Process(
            target=_run,
            args=(payload, request_queue, result_queue),
        )

        result: OptimizerCallbackResult | None
        exception: Exception | None = None

        _logger.info("Starting external optimization in subprocess")
        process.start()
        try:
            while exception is None:
                try:
                    request = request_queue.get(timeout=_QUEUE_POLL_INTERVAL)
                except queue.Empty:
                    if not process.is_alive():
                        _logger.warning(
                            "External backend subprocess died unexpectedly (exit code %s)",
                            process.exitcode,
                        )
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
                except Exception as exc:  # ruff: ignore[blind-except]
                    result = None
                    exception = exc
                result_queue.put(result)
        finally:
            _shutdown(process, request_queue, result_queue)

        if exception is not None:
            raise exception

    def validate_options(  # ruff: ignore[undocumented-public-method]
        self,
    ) -> None:
        self._backend_plugin.create(self._backend_config).validate_options()

    @property
    def is_parallel(self) -> bool:  # ruff: ignore[undocumented-public-method]
        return self._is_parallel

    def _serialize(self, initial_values: NDArray[np.float64]) -> bytes:
        # The context points back at this backend, which holds the optimizer
        # callback, so the entire ensemble graph would travel with it. The child
        # replaces the backend and the callback both, so that edge is cut here
        # rather than sent.
        #
        # Cut on the context itself, not on a copy: plugin instances store the
        # context in their `init`, so a copy leaves the original reachable
        # through them and both end up in the payload.
        backend = self._context.backend
        object.__setattr__(self._context, "backend", None)  # ruff: ignore[unnecessary-dunder-call]
        try:
            return dumps(
                {
                    "config": self._backend_config,
                    "context": self._context,
                    "initial_values": initial_values,
                }
            )
        except Exception as exc:
            msg = (
                "The optimization could not be sent to an external process: "
                f"{CANNOT_SERIALIZE}."
            )
            raise ExecutionError(msg) from exc
        finally:
            object.__setattr__(self._context, "backend", backend)  # ruff: ignore[unnecessary-dunder-call]


def _shutdown(
    process: BaseProcess,
    request_queue: multiprocessing.Queue[dict[str, Any] | None],
    result_queue: multiprocessing.Queue[OptimizerCallbackResult | None],
) -> None:
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
    request_queue.close()
    request_queue.join_thread()
    result_queue.close()
    result_queue.join_thread()


def _run(
    data: bytes,
    request_queue: multiprocessing.Queue[dict[str, Any] | None],
    result_queue: multiprocessing.Queue[OptimizerCallbackResult | None],
) -> None:
    try:
        backend, initial_values = _prepare(data, request_queue, result_queue)
        # Suppressed rather than handled, and deliberately not spanning
        # `_prepare`: this arm reports nothing while the `finally` queues the
        # sentinel either way, so letting it cover setup would turn a setup
        # failure into a run that ends with no error and no result.
        # `backend.start` is also the only place it is raised from.
        with contextlib.suppress(_DriverStopped):
            backend.start(initial_values)
    except OptimizerStop as exc:
        request_queue.put({"stop": True, "exit_code": exc.exit_code})
    except Exception as exc:  # ruff: ignore[blind-except]
        request_queue.put(_encode_child_exception(exc))
    finally:
        request_queue.put(None)


def _prepare(
    data: bytes,
    request_queue: multiprocessing.Queue[dict[str, Any] | None],
    result_queue: multiprocessing.Queue[OptimizerCallbackResult | None],
) -> tuple[Backend, NDArray[np.float64]]:
    # Workflow objects arrive as inert placeholders and are never used here.
    try:
        data_dict = loads(data)
    except Exception as exc:
        exc.add_note(f"Could not rebuild the optimization: {CANNOT_DESERIALIZE}.")
        raise

    config = data_dict["config"]
    backend = get_plugin("backend", method=config.method).create(config)
    context = data_dict["context"].model_copy(update={"backend": backend})
    backend.init(
        context,
        partial(_callback, request_queue=request_queue, result_queue=result_queue),
    )
    return backend, np.asarray(data_dict["initial_values"], dtype=np.float64)


def _callback(
    variables: NDArray[np.float64],
    *,
    return_functions: bool,
    return_gradients: bool,
    request_queue: multiprocessing.Queue[dict[str, Any] | None],
    result_queue: multiprocessing.Queue[OptimizerCallbackResult | None],
) -> OptimizerCallbackResult:
    request_queue.put(
        {
            "variables": variables,
            "return_functions": return_functions,
            "return_gradients": return_gradients,
        },
    )
    result = result_queue.get()
    if result is None:
        # The evaluation callback failed. The driver re-raises the exception.
        raise _DriverStopped
    return result


def _handle_request(
    request: dict[str, Any],
) -> Exception | dict[str, Any]:
    if "stop" in request:
        return OptimizerStop(request["exit_code"])
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
        pickled = dumps(exc)
    except Exception:  # ruff: ignore[blind-except]
        notes = "".join(f"\n{note}" for note in getattr(exc, "__notes__", []))
        return {
            "error": type(exc).__name__,
            "message": f"{exc}{notes}",
            "traceback": tb_str,
        }
    return {"exception": pickled, "traceback": tb_str}


def _decode_child_exception(request: dict[str, Any]) -> Exception:
    tb_str = request.get("traceback", "")
    try:
        original = loads(request["exception"])
    except Exception:  # ruff: ignore[blind-except]
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
