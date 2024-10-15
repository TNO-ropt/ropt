from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from ropt.config.enopt import EnOptConfig
from ropt.plan import OptimizationPlanRunner

# Requires pandas:
pd = pytest.importorskip("pandas")


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> Dict[str, Any]:
    return {
        "optimizer": {
            "tolerance": 1e-5,
            "speculative": True,
            "max_functions": 3,
        },
        "variables": {
            "initial_values": [0.0, 0.0, 0.1],
            "upper_bounds": 1.0,
            "lower_bounds": -1.0,
        },
        "objective_functions": {
            "weights": [0.75, 0.25],
        },
        "gradient": {
            "perturbation_magnitudes": 0.01,
        },
    }


def test_tabular_report_no_results(
    enopt_config: Any, evaluator: Any, tmp_path: Path
) -> None:
    path = tmp_path / "results.txt"
    OptimizationPlanRunner(enopt_config, evaluator()).add_table(
        {}, path=tmp_path / "results.txt"
    ).run()
    assert not path.exists()


def test_tabular_report_results(
    enopt_config: Any, evaluator: Any, tmp_path: Path
) -> None:
    path = tmp_path / "results.txt"
    config = EnOptConfig.model_validate(enopt_config)
    OptimizationPlanRunner(config, evaluator()).add_table(
        {
            "result_id": "eval-ID",
            "evaluations.variables": "Variables",
        },
        path=path,
    ).run()

    assert path.exists()
    results = pd.read_fwf(tmp_path / "results.txt", header=[0, 1], skiprows=[2])

    assert (
        results.columns.get_level_values(level=0).to_list()
        == ["eval-ID"] + ["Variables"] * 3
    )
    assert results.columns.get_level_values(level=1)[1:].to_list() == [
        "0",
        "1",
        "2",
    ]
    assert len(results) == 3


def test_tabular_report_data_frames_results_formatted_names(
    enopt_config: Any, evaluator: Any, tmp_path: Path
) -> None:
    path = tmp_path / "results.txt"
    enopt_config["variables"]["names"] = [("a", 1), ("a", 2), ("a", 3)]
    config = EnOptConfig.model_validate(enopt_config)
    OptimizationPlanRunner(config, evaluator()).add_table(
        {
            "result_id": "eval-ID",
            "evaluations.variables": "Variables",
        },
        path=path,
    ).run()

    assert path.exists()
    results = pd.read_fwf(path, header=[0, 1], skiprows=[2])
    assert results.columns.get_level_values(level=0)[1:].to_list() == ["Variables"] * 3
    assert list(results.columns.get_level_values(level=1)[1:].to_list()) == [
        f"a:{idx + 1}" for idx in range(3)
    ]


def test_tabular_report_data_frames_gradients(
    enopt_config: Any, evaluator: Any, tmp_path: Path
) -> None:
    path = tmp_path / "gradients.txt"
    enopt_config["variables"]["names"] = [("a", 1), ("a", 2), ("a", 3)]
    config = EnOptConfig.model_validate(enopt_config)
    OptimizationPlanRunner(config, evaluator()).add_table(
        {
            "result_id": "eval-ID",
            "gradients.weighted_objective": "Total Objective",
        },
        path,
        table_type="gradients",
    ).run()
    assert path.exists()
    gradients = pd.read_fwf(path, header=[0, 1], skiprows=[2])
    assert (
        gradients.columns.get_level_values(level=0).to_list()
        == ["eval-ID"] + ["Total Objective"] * 3
    )
    assert gradients.columns.get_level_values(level=1)[1:].to_list() == [
        "a:1",
        "a:2",
        "a:3",
    ]
    assert len(gradients) == 3


@pytest.mark.parametrize("min_header_len", [None, 4])
def test_tabular_report_data_frames_min_header_len(
    enopt_config: Any, evaluator: Any, tmp_path: Path, min_header_len: Optional[int]
) -> None:
    path = tmp_path / "results.txt"
    enopt_config["variables"]["names"] = [("a", 1), ("a", 2), ("a", 3)]
    config = EnOptConfig.model_validate(enopt_config)
    OptimizationPlanRunner(config, evaluator()).add_table(
        {
            "result_id": "eval-ID",
            "evaluations.variables": "Variables",
        },
        path=path,
        min_header_len=min_header_len,
    ).run()

    assert path.exists()
    with path.open() as fp:
        lines = fp.readlines()
    header_idx = 2 if min_header_len is None else min_header_len
    assert lines[header_idx][0] == "-"
    results = pd.read_fwf(
        path,
        header=list(range(header_idx)),
        skip_blank_lines=False,
        skiprows=[header_idx],
    )
    assert len(results.columns[0]) == header_idx
    assert len(results) == 3
