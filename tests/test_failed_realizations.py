import numpy as np

from ropt.core._evaluator import (
    _get_failed_function_realizations,
    _get_failed_gradient_realizations,
)


def test_failed_function_realization_nan_objective() -> None:
    failed_realizations = _get_failed_function_realizations(np.array([[np.nan, 1.0]]))
    assert np.all(failed_realizations == [True])


def test_failed_gradient_realization_no_perturbation_success() -> None:
    failed_realizations = _get_failed_gradient_realizations(
        np.array([[1.0, 1.0]]), np.array([[5 * [np.nan, 1.0]]]), 1
    )
    assert np.all(failed_realizations == [True])


def test_failed_gradient_realization_perturbation_successes_below_min() -> None:
    failed_realizations = _get_failed_gradient_realizations(
        np.array([[1.0, 1.0]]),
        np.array(
            [[[1.0, 1.0], [1.0, 1.0], [np.nan, 1.0], [np.nan, 1.0], [np.nan, np.nan]]]
        ),
        4,
    )
    assert np.all(failed_realizations == [True])


def test_gradient_realization_perturbation_successes_meet_min() -> None:
    failed_realizations = _get_failed_gradient_realizations(
        np.array([[1.0, 1.0]]),
        np.array(
            [[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [np.nan, 1.0]]],
        ),
        4,
    )
    assert np.all(failed_realizations == [False])


def test_failed_gradient_realization_per_realization() -> None:
    failed_realizations = _get_failed_gradient_realizations(
        np.array([[1.0, 1.0], [1.0, 1.0]]),
        np.array([[[1.0, 1.0]], [[np.nan, 1.0]]]),
        1,
    )
    assert np.all(failed_realizations == [False, True])


def test_failed_gradient_realization_nan_objective() -> None:
    failed_realizations = _get_failed_gradient_realizations(
        np.array([[1.0, 1.0], [np.nan, 1.0]]),
        np.array([[[1.0], [1.0]], [[1.0], [1.0]]]),
        1,
    )
    assert np.all(failed_realizations == [False, True])


def test_failed_gradient_realization_all_nan_objective_entries() -> None:
    failed_realizations = _get_failed_gradient_realizations(
        np.array([[1.0, 1.0], [np.nan, np.nan]]),
        np.array([[[1.0], [1.0]], [[1.0], [1.0]]]),
        1,
    )
    assert np.all(failed_realizations == [False, True])


def test_gradient_realization_single_perturbation_success() -> None:
    failed_realizations = _get_failed_gradient_realizations(
        np.array([[1.0, 1.0], [1.0, 1.0]]),
        np.array([[[1.0, 1.0], [1.0, 1.0]], [[np.nan, np.nan], [1.0, 1.0]]]),
        1,
    )
    assert np.all(failed_realizations == [False, False])
