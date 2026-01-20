from __future__ import annotations

import asyncio
from functools import partial
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import pytest

from ropt.config import EnOptConfig
from ropt.plugins import PluginManager
from ropt.plugins.server.base import PollingServer, Server, ServerPlugin, Task
from ropt.workflow import (
    create_compute_step,
    create_evaluator,
    create_event_handler,
    create_server,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

    from ropt.results import FunctionResults


pytestmark = [pytest.mark.asyncio, pytest.mark.timeout(1)]


initial_values = np.array([0.0, 0.0, 0.1])


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> dict[str, Any]:
    return {
        "optimizer": {
            "tolerance": 1e-5,
            "max_functions": 20,
        },
        "variables": {
            "variable_count": len(initial_values),
            "perturbation_magnitudes": 0.01,
        },
        "objectives": {
            "weights": [0.75, 0.25],
        },
    }


def _function(
    variables: NDArray[np.float64],
    realization: int,
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


class MockedPollingServer(PollingServer[R, TR]):
    def __init__(self, *, workers: int = 1, maxsize: int = 0) -> None:
        super().__init__(workers=workers, maxsize=maxsize, interval=0.0)
        self._results: dict[Task[R, TR], R] = {}

    def submit(self, task: Task[R, TR]) -> None:
        self._results[task] = task.function()

    def poll(self) -> None:
        for task in self._tasks:
            if task in self._results:
                self._tasks[task] = "done"

    def get_result(self, task: Task[R, TR]) -> R:
        return self._results.pop(task)


class MockedServerPlugin(ServerPlugin):
    @classmethod
    def create(
        cls,
        name: str,  # noqa: ARG003
        **kwargs: Any,
    ) -> Server:
        return MockedPollingServer(**kwargs)

    @classmethod
    def is_supported(cls, method: str) -> bool:
        return method.lower() == "polling_test"


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
    "server_name", ["async_server", "multiprocessing_server", "polling_test"]
)
async def test_async_evaluator_ok(
    enopt_config: dict[str, Any],
    test_functions: Sequence[Callable[[NDArray[np.float64], int], float]],
    server_name: str,
    monkeypatch: Any,
) -> None:
    if server_name == "polling_test":
        manager = PluginManager()
        monkeypatch.setattr(manager, "_init", lambda: None)
        manager._add_plugin("server", "polling_test", MockedServerPlugin)  # noqa: SLF001
        server = manager.get_plugin("server", "polling_test").create("polling_test")
    else:
        server = create_server(server_name, workers=2)
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
    "server_name", ["async_server", "multiprocessing_server", "polling_test"]
)
async def test_async_evaluator_error(
    enopt_config: dict[str, Any],
    test_functions: Sequence[Callable[[NDArray[np.float64], int], float]],
    server_name: str,
    monkeypatch: Any,
) -> None:
    if server_name == "polling_test":
        manager = PluginManager()
        monkeypatch.setattr(manager, "_init", lambda: None)
        manager._add_plugin("server", "polling_test", MockedServerPlugin)  # noqa: SLF001
        server = manager.get_plugin("server", "polling_test").create("polling_test")
    else:
        server = create_server(server_name, workers=2)
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
    "server_name", ["async_server", "multiprocessing_server", "polling_test"]
)
async def test_async_evaluator_two_optimizations(
    enopt_config: dict[str, Any],
    test_functions: Sequence[Callable[[NDArray[np.float64], int], float]],
    server_name: str,
    monkeypatch: Any,
) -> None:
    if server_name == "polling_test":
        manager = PluginManager()
        monkeypatch.setattr(manager, "_init", lambda: None)
        manager._add_plugin("server", "polling_test", MockedServerPlugin)  # noqa: SLF001
        server = manager.get_plugin("server", "polling_test").create("polling_test")
    else:
        server = create_server(server_name, workers=2)
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
