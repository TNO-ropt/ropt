from pathlib import Path
from typing import Any, Dict

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
def enopt_config_fixture() -> Dict[str, Any]:
    return {
        "variables": {
            "names": ["x", "y"],
            "initial_values": [0.0, 0.0],
        },
        "objective_functions": {
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
        result_id=0,
        batch_id=1,
        metadata={},
        evaluations=evaluations,
        realizations=realizations,
        functions=functions,
    )


@pytest.fixture(name="gradient_result")
def gradient_result_fixture() -> GradientResults:
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
        result_id=0,
        batch_id=1,
        metadata={},
        evaluations=evaluations,
        realizations=Realizations(
            failed_realizations=np.zeros(3, dtype=np.bool_),
        ),
        gradients=gradients,
    )


@pytest.mark.parametrize(
    ("name", "final_name"),
    [
        ("test", "test.nc"),
        ("test.nc", "test.nc"),
        ("test{batch_id:03d}", "test001.nc"),
        ("test{batch_id:03d}.nc", "test001.nc"),
    ],
)
def test_functions_to_netcdf(
    enopt_config: Any,
    function_result: FunctionResults,
    tmp_path: Path,
    name: str,
    final_name: str,
) -> None:
    config = EnOptConfig.model_validate(enopt_config)
    function_result.to_netcdf(config, tmp_path / name)
    assert (tmp_path / final_name).exists()
    file = netcdf.Dataset((tmp_path / final_name), mode="r")
    assert set(file.groups) == {
        "evaluations",
        "realizations",
        "functions",
        "__metadata__",
    }


def test_functions_from_netcdf(
    enopt_config: Any, function_result: FunctionResults, tmp_path: Path
) -> None:
    config = EnOptConfig.model_validate(enopt_config)
    function_result.to_netcdf(config, tmp_path / "test.nc")
    assert (tmp_path / "test.nc").exists()
    loaded = FunctionResults.from_netcdf(tmp_path / "test.nc")
    assert loaded.batch_id == function_result.batch_id
    for item in ("evaluations", "realizations", "functions"):
        assert loaded.to_dataframe(config, item).equals(
            function_result.to_dataframe(config, item)
        )


@pytest.mark.parametrize(
    ("name", "final_name"),
    [
        ("test", "test.nc"),
        ("test.nc", "test.nc"),
        ("test{batch_id:03d}", "test001.nc"),
        ("test{batch_id:03d}.nc", "test001.nc"),
    ],
)
def test_gradients_to_netcdf(
    enopt_config: Any,
    gradient_result: GradientResults,
    tmp_path: Path,
    name: str,
    final_name: str,
) -> None:
    config = EnOptConfig.model_validate(enopt_config)
    gradient_result.to_netcdf(config, tmp_path / name)
    assert (tmp_path / final_name).exists()
    file = netcdf.Dataset((tmp_path / final_name), mode="r")
    assert set(file.groups) == {
        "evaluations",
        "realizations",
        "gradients",
        "__metadata__",
    }


def test_gradients_from_netcdf(
    enopt_config: Any, gradient_result: GradientResults, tmp_path: Path
) -> None:
    config = EnOptConfig.model_validate(enopt_config)
    gradient_result.to_netcdf(config, tmp_path / "test.nc")
    assert (tmp_path / "test.nc").exists()
    loaded = GradientResults.from_netcdf(tmp_path / "test.nc")
    assert loaded.batch_id == gradient_result.batch_id
    for item in ("evaluations", "gradients"):
        assert loaded.to_dataframe(config, item).equals(
            gradient_result.to_dataframe(config, item)
        )


def test_netcdf_metadata(
    enopt_config: Any, function_result: FunctionResults, tmp_path: Path
) -> None:
    config = EnOptConfig.model_validate(enopt_config)
    function_result.metadata = {"foo": 1}
    function_result.to_netcdf(config, tmp_path / "test.nc")
    assert (tmp_path / "test.nc").exists()
    loaded = FunctionResults.from_netcdf(tmp_path / "test.nc")
    assert loaded.metadata == {"foo": 1}
