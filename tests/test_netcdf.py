from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ropt.config.enopt import EnOptConfig
from ropt.results import (
    FunctionEvaluations,
    FunctionResults,
    Functions,
    GradientEvaluations,
    GradientResults,
    Gradients,
    Realizations,
)

pytest.importorskip("xarray")
netcdf = pytest.importorskip("netCDF4")


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> dict[str, Any]:
    return {
        "variables": {
            "names": ["x", "y"],
            "initial_values": [0.0, 0.0],
        },
        "objectives": {
            "names": ["f1", "f2"],
            "weights": [0.75, 0.25],
        },
        "realizations": {
            "names": ["r1", "r2", "r3"],
            "weights": [1.0] * 3,
        },
        "gradient": {
            "number_of_perturbations": 5,
        },
    }


@pytest.fixture(name="function_result")
def function_result_fixture(enopt_config: Any) -> FunctionResults:
    config = EnOptConfig.model_validate(enopt_config)
    evaluations = FunctionEvaluations.create(
        config=config,
        variables=np.array([1.0, 2.0]),
        objectives=np.arange(6, dtype=np.float64).reshape((3, 2)),
    )
    realizations = Realizations(
        objective_weights=np.arange(6, dtype=np.float64).reshape((2, 3)),
        failed_realizations=np.zeros(3, dtype=np.bool_),
    )
    functions = Functions.create(
        config=config,
        weighted_objective=np.array(1.0),
        objectives=np.array([1.0, 2.0]),
    )
    return FunctionResults(
        plan_id=(0,),
        result_id=0,
        batch_id=1,
        metadata={},
        config=config,
        evaluations=evaluations,
        realizations=realizations,
        functions=functions,
    )


@pytest.fixture(name="gradient_result")
def gradient_result_fixture(enopt_config: Any) -> GradientResults:
    config = EnOptConfig.model_validate(enopt_config)
    evaluations = GradientEvaluations(
        variables=np.array([1.0, 2.0]),
        perturbed_variables=np.arange(30, dtype=np.float64).reshape((3, 5, 2)),
        perturbed_objectives=np.arange(30, dtype=np.float64).reshape((3, 5, 2)),
    )
    gradients = Gradients(
        weighted_objective=np.array([1.0, 2.0]),
        objectives=np.arange(4, dtype=np.float64).reshape((2, 2)),
    )
    return GradientResults(
        plan_id=(0,),
        result_id=0,
        batch_id=1,
        metadata={},
        config=config,
        evaluations=evaluations,
        realizations=Realizations(
            failed_realizations=np.zeros(3, dtype=np.bool_),
        ),
        gradients=gradients,
    )


def test_functions_to_netcdf(function_result: FunctionResults, tmp_path: Path) -> None:
    function_result.to_netcdf(tmp_path / "test.nc")
    assert (tmp_path / "test.nc").exists()
    file = netcdf.Dataset((tmp_path / "test.nc"), mode="r")
    assert set(file.groups) == {
        "evaluations",
        "realizations",
        "functions",
        "__metadata__",
    }


def test_functions_from_netcdf(
    function_result: FunctionResults, tmp_path: Path
) -> None:
    function_result.to_netcdf(tmp_path / "test.nc")
    assert (tmp_path / "test.nc").exists()
    loaded = FunctionResults.from_netcdf(tmp_path / "test.nc")
    assert loaded.result_id == function_result.result_id
    assert loaded.batch_id == function_result.batch_id
    assert loaded.plan_id == function_result.plan_id
    for item in ("evaluations", "realizations", "functions"):
        assert loaded.to_dataframe(item).equals(function_result.to_dataframe(item))


def test_gradients_to_netcdf(gradient_result: GradientResults, tmp_path: Path) -> None:
    gradient_result.to_netcdf(tmp_path / "test.nc")
    assert (tmp_path / "test.nc").exists()
    file = netcdf.Dataset((tmp_path / "test.nc"), mode="r")
    assert set(file.groups) == {
        "evaluations",
        "realizations",
        "gradients",
        "__metadata__",
    }


def test_gradients_from_netcdf(
    gradient_result: GradientResults, tmp_path: Path
) -> None:
    gradient_result.to_netcdf(tmp_path / "test.nc")
    assert (tmp_path / "test.nc").exists()
    loaded = GradientResults.from_netcdf(tmp_path / "test.nc")
    assert loaded.batch_id == gradient_result.batch_id
    for item in ("evaluations", "gradients"):
        assert loaded.to_dataframe(item).equals(gradient_result.to_dataframe(item))


def test_netcdf_metadata(function_result: FunctionResults, tmp_path: Path) -> None:
    function_result.metadata = {"foo": 1}
    function_result.to_netcdf(tmp_path / "test.nc")
    assert (tmp_path / "test.nc").exists()
    loaded = FunctionResults.from_netcdf(tmp_path / "test.nc")
    assert loaded.metadata == {"foo": 1}
