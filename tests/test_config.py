import copy
from typing import Any

import numpy as np
import pytest

from ropt.config.enopt import EnOptConfig, GradientConfig, LinearConstraintsConfig
from ropt.enums import BoundaryType, PerturbationType
from ropt.transforms import OptModelTransforms, VariableScaler


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> dict[str, Any]:
    return {
        "variables": {
            "initial_values": np.array([1, 2]),
        },
        "objectives": {
            "weights": [1.0],
        },
        "optimizer": {
            "method": "dummy",
        },
    }


def test_check_linear_constraints() -> None:
    config = {
        "coefficients": np.array([[1, 2], [1, 2], [1, 2]]),
        "lower_bounds": np.array([-np.inf, 2, 3]),
        "upper_bounds": np.array([1, np.inf, 3]),
    }
    linear_constraints = LinearConstraintsConfig.model_validate(config)
    assert linear_constraints.coefficients is not None
    with pytest.raises(ValueError):  # noqa: PT011
        linear_constraints.coefficients[0, 0] = 0
    with pytest.raises(ValueError):  # noqa: PT011
        linear_constraints.upper_bounds[0] = 1


def test_check_linear_constraints_convert() -> None:
    config = {
        "coefficients": [[1, 2], [1, 2], [1, 2]],
        "lower_bounds": np.array([-np.inf, 2, 3]),
        "upper_bounds": np.array([1, np.inf, 3]),
    }
    LinearConstraintsConfig.model_validate(config)


def test_check_linear_constraints_vector_shapes() -> None:
    config = {
        "coefficients": [[1, 2, 3], [1, 2, 3]],
        "lower_bounds": np.array([-np.inf, 2]),
        "upper_bounds": np.array([1, np.inf]),
    }
    LinearConstraintsConfig.model_validate(config)

    config_copy = copy.deepcopy(config)
    config_copy["lower_bounds"] = [1, 2, 3]
    with pytest.raises(
        ValueError,
        match="lower_bounds cannot be broadcasted to a length of 2",
    ):
        LinearConstraintsConfig.model_validate(config_copy)


def test_check_perturbations() -> None:
    GradientConfig()
    gradients = GradientConfig(perturbation_magnitudes=np.array([0.1]))
    assert gradients.perturbation_magnitudes == np.array([0.1])


def test_check_config(enopt_config: Any) -> None:
    EnOptConfig.model_validate(enopt_config)


def test_check_config_linear_constraints(enopt_config: Any) -> None:
    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 2, 3], [2, 3, 4]],
        "lower_bounds": [1, 2],
        "upper_bounds": [np.inf, np.inf],
    }
    with pytest.raises(
        ValueError,
        match="the coefficients matrix should have 2 columns",
    ):
        EnOptConfig.model_validate(enopt_config)


def test_check_config_perturbations(enopt_config: Any) -> None:
    enopt_config["gradient"] = {
        "perturbation_magnitudes": [1] * 2,
        "boundary_types": [BoundaryType.TRUNCATE_BOTH] * 2,
        "perturbation_types": [PerturbationType.ABSOLUTE] * 2,
    }
    EnOptConfig.model_validate(enopt_config)

    config_copy = copy.deepcopy(enopt_config)
    config_copy["gradient"]["perturbation_magnitudes"] = [1] * 3
    with pytest.raises(
        ValueError,
        match="the perturbation magnitudes cannot be broadcasted to a length of 2",
    ):
        EnOptConfig.model_validate(config_copy)

    config_copy = copy.deepcopy(enopt_config)
    config_copy["gradient"]["boundary_types"] = [BoundaryType.TRUNCATE_BOTH] * 3
    with pytest.raises(
        ValueError, match="perturbation boundary_types must have 2 items"
    ):
        EnOptConfig.model_validate(config_copy)

    config_copy = copy.deepcopy(enopt_config)
    config_copy["gradient"]["perturbation_types"] = [PerturbationType.ABSOLUTE] * 3
    with pytest.raises(ValueError, match="perturbation types must have 2 items"):
        EnOptConfig.model_validate(config_copy)


def test_check_config_min_success(enopt_config: Any) -> None:
    def gen_config(pert_min: int | None, real_min: int | None) -> dict[str, Any]:
        config: dict[str, Any] = copy.deepcopy(enopt_config)
        config["realizations"] = {"weights": 4 * [1.0]}
        config["gradient"] = {}
        if pert_min is not None:
            config["gradient"]["perturbation_min_success"] = pert_min
        if real_min is not None:
            config["realizations"]["realization_min_success"] = real_min
        return config

    pert_test_map = {None: 5, 1: 1, 4: 4, 7: 5}
    real_test_map = {None: 4, 1: 1, 3: 3, 7: 4}
    test_space = zip(pert_test_map.keys(), real_test_map.keys(), strict=False)
    for pert_in, real_in in test_space:
        config = EnOptConfig.model_validate(gen_config(pert_in, real_in))
        assert pert_test_map[pert_in] == config.gradient.perturbation_min_success
        assert real_test_map[real_in] == config.realizations.realization_min_success


def test_perturbation_types(enopt_config: Any) -> None:
    enopt_config["gradient"] = {
        "perturbation_magnitudes": [0.1, 0.01],
        "perturbation_types": [PerturbationType.ABSOLUTE, PerturbationType.RELATIVE],
    }
    enopt_config["variables"]["lower_bounds"] = [0.0, 600]
    enopt_config["variables"]["upper_bounds"] = [1.0, np.inf]
    with pytest.raises(
        ValueError,
        match="The variable bounds must be finite to use relative perturbations",
    ):
        config = EnOptConfig.model_validate(enopt_config)

    enopt_config["variables"]["initial_values"] = [0.0, 0.0, 0.0]
    enopt_config["variables"]["lower_bounds"] = [0.0, 100.0, 0.0]
    enopt_config["variables"]["upper_bounds"] = [np.inf, 600.0, 1.0]
    enopt_config["gradient"] = {
        "perturbation_magnitudes": [0.1, 0.01, 1.0],
        "perturbation_types": [
            PerturbationType.ABSOLUTE,
            PerturbationType.RELATIVE,
            PerturbationType.ABSOLUTE,
        ],
    }
    config = EnOptConfig.model_validate(enopt_config)
    assert np.allclose(config.gradient.perturbation_magnitudes, [0.1, 5.0, 1.0])


def test_perturbation_types_with_scaler(enopt_config: Any) -> None:
    enopt_config["gradient"] = {
        "perturbation_magnitudes": [0.1, 0.01],
        "perturbation_types": [PerturbationType.ABSOLUTE, PerturbationType.RELATIVE],
    }
    enopt_config["variables"]["lower_bounds"] = [0.0, 600]
    enopt_config["variables"]["upper_bounds"] = [1.0, np.inf]
    with pytest.raises(
        ValueError,
        match="The variable bounds must be finite to use relative perturbations",
    ):
        config = EnOptConfig.model_validate(enopt_config)

    enopt_config["variables"]["initial_values"] = [0.0, 0.0, 0.0]
    enopt_config["variables"]["lower_bounds"] = [0.0, 100.0, 0.0]
    enopt_config["variables"]["upper_bounds"] = [np.inf, 600.0, 1.0]

    enopt_config["gradient"] = {
        "perturbation_magnitudes": [0.1, 0.01, 1.0],
        "perturbation_types": [
            PerturbationType.ABSOLUTE,
            PerturbationType.RELATIVE,
            PerturbationType.ABSOLUTE,
        ],
    }
    config = EnOptConfig.model_validate(
        enopt_config,
        context=OptModelTransforms(
            variables=VariableScaler(np.array([1.0, 1.0, 50.0]), None)
        ),
    )
    assert np.allclose(config.gradient.perturbation_magnitudes, [0.1, 5.0, 0.02])
