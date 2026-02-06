from __future__ import annotations

import asyncio
import subprocess  # noqa: S404
from functools import partial
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np
import pytest

from ropt.config import EnOptConfig
from ropt.workflow import (
    create_compute_step,
    create_evaluator,
    create_event_handler,
    create_server,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path
    from uuid import UUID

    from numpy.typing import NDArray

    from ropt.plugins.server.base import RemoteTaskState, Server, Task
    from ropt.results import FunctionResults

try:
    import cloudpickle

    HAS_CLOUDPICKLE = True
except ImportError:
    HAS_CLOUDPICKLE = False

pytestmark = [pytest.mark.asyncio, pytest.mark.timeout(1)]


initial_values = np.array([0.0, 0.0, 0.1])


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> dict[str, Any]:
    return {
        "optimizer": {
            "tolerance": 1e-2,
            "max_functions": 8,
        },
        "variables": {
            "variable_count": len(initial_values),
            "perturbation_magnitudes": 0.001,
        },
        "gradient": {
            "number_of_perturbations": 3,
        },
        "objectives": {
            "weights": [0.75, 0.25],
        },
    }


def _function(
    variables: NDArray[np.float64],
    realization: int,
    perturbation: int,  # noqa: ARG001
    test_functions: Sequence[Callable[[NDArray[np.float64], int], float]],
    *,
    raise_error: bool = False,
) -> NDArray[np.float64]:
    if raise_error:
        msg = "Test error in function"
        raise ValueError(msg)
    return np.fromiter(
        (func(variables, realization) for func in test_functions), dtype=np.float64
    )


R = TypeVar("R")
TR = TypeVar("TR")


class MockedBasicRemoteAdapter(Generic[R, TR]):
    def __init__(self) -> None:
        self._results: dict[UUID, R] = {}

    def submit(self, task: Task[R, TR]) -> None:
        self._results[task.id] = task.function(*task.args)

    def poll(self) -> dict[UUID, RemoteTaskState]:
        return dict.fromkeys(self._results, "done")

    def get_result(self, task_id: UUID) -> R | Exception:
        return self._results.pop(task_id)


def _workflow(
    server: Server,
    enopt_config: dict[str, Any],
    test_function: Callable[[NDArray[np.float64], int], NDArray[np.float64]],
) -> FunctionResults:
    evaluator = create_evaluator(
        "async_evaluator", function=test_function, server=server
    )
    step = create_compute_step("optimizer", evaluator=evaluator)
    tracker = create_event_handler("tracker")
    step.add_event_handler(tracker)
    step.run(variables=initial_values, config=EnOptConfig.model_validate(enopt_config))
    results: FunctionResults = tracker["results"]
    return results


@pytest.mark.parametrize(
    "server_name", ["async_server", "multiprocessing_server", "remote_server"]
)
async def test_async_evaluator_ok(
    enopt_config: dict[str, Any],
    test_functions: Sequence[Callable[[NDArray[np.float64], int], float]],
    server_name: str,
) -> None:
    server = (
        create_server(
            "remote_server", remote=MockedBasicRemoteAdapter(), workers=2, interval=0.0
        )
        if server_name == "remote_server"
        else create_server(server_name, workers=2)
    )
    assert not server.is_running()
    async with asyncio.TaskGroup() as tg:
        await server.start(tg)
        assert server.is_running()
        results = await asyncio.to_thread(
            _workflow,
            server,
            enopt_config,
            partial(_function, test_functions=test_functions),
        )
        server.cancel()
    assert not server.is_running()

    assert results is not None
    assert np.allclose(results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)


@pytest.mark.parametrize(
    "server_name", ["async_server", "multiprocessing_server", "remote_server"]
)
async def test_async_evaluator_error(
    enopt_config: dict[str, Any],
    test_functions: Sequence[Callable[[NDArray[np.float64], int], float]],
    server_name: str,
) -> None:
    server = (
        create_server(
            "remote_server", remote=MockedBasicRemoteAdapter(), workers=2, interval=0.0
        )
        if server_name == "remote_server"
        else create_server(server_name, workers=2)
    )
    assert not server.is_running()
    with pytest.RaisesGroup(
        pytest.RaisesExc(ValueError, match="Test error in function")
    ):
        async with asyncio.TaskGroup() as tg:
            await server.start(tg)
            assert server.is_running()
            await asyncio.to_thread(
                _workflow,
                server,
                enopt_config,
                partial(_function, test_functions=test_functions, raise_error=True),
            )
            server.cancel()
    assert not server.is_running()


@pytest.mark.parametrize(
    "server_name", ["async_server", "multiprocessing_server", "remote_server"]
)
async def test_async_evaluator_two_optimizations(
    enopt_config: dict[str, Any],
    test_functions: Sequence[Callable[[NDArray[np.float64], int], float]],
    server_name: str,
) -> None:
    server = (
        create_server(
            "remote_server", remote=MockedBasicRemoteAdapter(), workers=2, interval=0.0
        )
        if server_name == "remote_server"
        else create_server(server_name, workers=2)
    )
    assert not server.is_running()
    async with asyncio.TaskGroup() as tg:
        await server.start(tg)
        assert server.is_running()
        results_list = await asyncio.gather(
            *(
                asyncio.to_thread(
                    _workflow,
                    server,
                    enopt_config,
                    partial(_function, test_functions=test_functions),
                )
                for _ in range(2)
            )
        )
        server.cancel()
    assert not server.is_running()

    assert len(results_list) == 2
    for results in results_list:
        assert results is not None
        assert np.allclose(results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)


if HAS_CLOUDPICKLE:

    class MockedCloudpickleRemoteAdapter(Generic[R, TR]):
        def __init__(self, workdir: Path) -> None:
            self._workdir = workdir
            self._results: dict[UUID, R | Exception] = {}
            self._processes: dict[UUID, subprocess.Popen[bytes]] = {}

        def submit(self, task: Task[R, TR]) -> None:
            input_file = self._workdir / f"{task.id}.in"
            output_file = self._workdir / f"{task.id}.out"
            with input_file.open("wb") as fp:
                cloudpickle.dump((task.function, task.args), fp)
            process = subprocess.Popen(  # noqa: S603
                [  # noqa: S607
                    "python",
                    "-m",
                    "ropt.plugins.server",
                    str(input_file),
                    str(output_file),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._processes[task.id] = process

        def poll(self) -> dict[UUID, RemoteTaskState]:
            remove = []
            for task_id, process in self._processes.items():
                if process.poll() is not None:
                    input_file = self._workdir / f"{task_id}.in"
                    output_file = self._workdir / f"{task_id}.out"
                    with output_file.open("rb") as fp:
                        self._results[task_id] = cloudpickle.load(fp)
                    input_file.unlink()
                    output_file.unlink()
                    remove.append(task_id)
            for task_id in remove:
                self._processes.pop(task_id)
            states: dict[UUID, RemoteTaskState] = {
                task_id: ("error" if isinstance(result, Exception) else "done")
                for (task_id, result) in self._results.items()
            }
            states.update(dict.fromkeys(self._processes, "running"))
            return states

        def get_result(self, task_id: UUID) -> R | Exception:
            return self._results.pop(task_id)


@pytest.mark.slow
@pytest.mark.timeout(30)
@pytest.mark.skipif(not HAS_CLOUDPICKLE, reason="cloudpickle is not installed")
async def test_async_evaluator_cloudpickle_ok(
    enopt_config: dict[str, Any],
    test_functions: Sequence[Callable[[NDArray[np.float64], int], float]],
    tmp_path: Path,
) -> None:
    server = create_server(
        "remote_server",
        remote=MockedCloudpickleRemoteAdapter(tmp_path),
        workers=2,
        interval=0.1,
    )
    assert not server.is_running()
    async with asyncio.TaskGroup() as tg:
        await server.start(tg)
        assert server.is_running()
        results = await asyncio.to_thread(
            _workflow,
            server,
            enopt_config,
            partial(_function, test_functions=test_functions),
        )
        server.cancel()
    assert not server.is_running()

    assert results is not None
    assert np.allclose(results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)


@pytest.mark.slow
@pytest.mark.timeout(30)
@pytest.mark.skipif(not HAS_CLOUDPICKLE, reason="cloudpickle is not installed")
async def test_async_evaluator_cloudpickle_error(
    enopt_config: dict[str, Any],
    test_functions: Sequence[Callable[[NDArray[np.float64], int], float]],
    tmp_path: Path,
) -> None:
    server = create_server(
        "remote_server",
        remote=MockedCloudpickleRemoteAdapter(tmp_path),
        workers=2,
        interval=0.1,
    )
    assert not server.is_running()
    with pytest.RaisesGroup(
        pytest.RaisesExc(ValueError, match="Test error in function")
    ):
        async with asyncio.TaskGroup() as tg:
            await server.start(tg)
            assert server.is_running()
            await asyncio.to_thread(
                _workflow,
                server,
                enopt_config,
                partial(_function, test_functions=test_functions, raise_error=True),
            )
            server.cancel()
    assert not server.is_running()
