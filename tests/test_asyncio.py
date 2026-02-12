from __future__ import annotations

import asyncio
import subprocess  # noqa: S404
from functools import partial
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import pytest

from ropt.config import EnOptConfig
from ropt.plugins.server._hpc_server import DefaultHPCServer
from ropt.workflow import (
    create_compute_step,
    create_evaluator,
    create_event_handler,
    create_server,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from numpy.typing import NDArray

    from ropt.plugins.server.base import Server
    from ropt.results import FunctionResults

try:
    import cloudpickle  # noqa: F401
    import pandas as pd
    import pysqa  # noqa: F401

    _TEST_HPC = True
except ImportError:
    _TEST_HPC = False


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


def _function(  # noqa: PLR0917
    variables: NDArray[np.float64],
    realization: int,
    perturbation: int,  # noqa: ARG001
    batch_id: int,  # noqa: ARG001
    eval_idx: int,  # noqa: ARG001
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


if _TEST_HPC:

    class MockedHPCAdapter:
        def __init__(self, path: Path) -> None:
            self._path = path
            self._jobs: dict[int, str] = {}
            self._job_id = 0

        def submit_job(self, job_name: str, command: str, **kwargs: Any) -> int:  # noqa: ARG002
            subprocess.Popen(command.split())  # noqa: S603
            self._job_id += 1
            self._jobs[self._job_id] = job_name
            return self._job_id

        def get_status_of_my_jobs(self) -> pd.DataFrame:
            running = [
                job_id
                for job_id, job_name in self._jobs.items()
                if not (self._path / f"{job_name}.out").exists()
            ]
            self._jobs = {job_id: self._jobs[job_id] for job_id in running}
            return pd.DataFrame(list(self._jobs.keys()), columns=["jobid"])


@pytest.mark.parametrize(
    "server_name",
    [
        "async_server",
        "multiprocessing_server",
        pytest.param(
            "hpc_server",
            marks=[
                pytest.mark.slow,
                pytest.mark.timeout(30),
                pytest.mark.skipif(
                    not _TEST_HPC, reason="hpc requirements are not installed"
                ),
            ],
        ),
    ],
)
async def test_async_evaluator_ok(
    enopt_config: dict[str, Any],
    test_functions: Sequence[Callable[[NDArray[np.float64], int], float]],
    server_name: str,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    if server_name == "hpc_server":
        monkeypatch.setattr(
            "ropt.plugins.server._hpc_server.pysqa.QueueAdapter",
            lambda *args, **kwargs: MockedHPCAdapter(tmp_path),  # noqa: ARG005
        )
        server: Server = DefaultHPCServer(
            workdir=tmp_path, workers=2, interval=0, template=""
        )
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
    "server_name",
    [
        "async_server",
        "multiprocessing_server",
        pytest.param(
            "hpc_server",
            marks=[
                pytest.mark.slow,
                pytest.mark.timeout(30),
                pytest.mark.skipif(
                    not _TEST_HPC, reason="hpc requirements are not installed"
                ),
            ],
        ),
    ],
)
async def test_async_evaluator_error(
    enopt_config: dict[str, Any],
    test_functions: Sequence[Callable[[NDArray[np.float64], int], float]],
    server_name: str,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    if server_name == "hpc_server":
        monkeypatch.setattr(
            "ropt.plugins.server._hpc_server.pysqa.QueueAdapter",
            lambda *args, **kwargs: MockedHPCAdapter(tmp_path),  # noqa: ARG005
        )
        server: Server = DefaultHPCServer(
            workdir=tmp_path, workers=2, interval=0, template=""
        )
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
    "server_name",
    [
        "async_server",
        "multiprocessing_server",
        pytest.param(
            "hpc_server",
            marks=[
                pytest.mark.slow,
                pytest.mark.timeout(30),
                pytest.mark.skipif(
                    not _TEST_HPC, reason="hpc requirements are not installed"
                ),
            ],
        ),
    ],
)
async def test_async_evaluator_two_optimizations(
    enopt_config: dict[str, Any],
    test_functions: Sequence[Callable[[NDArray[np.float64], int], float]],
    server_name: str,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    if server_name == "hpc_server":
        monkeypatch.setattr(
            "ropt.plugins.server._hpc_server.pysqa.QueueAdapter",
            lambda *args, **kwargs: MockedHPCAdapter(tmp_path),  # noqa: ARG005
        )
        server: Server = DefaultHPCServer(
            workdir=tmp_path, workers=2, interval=0, template=""
        )
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
