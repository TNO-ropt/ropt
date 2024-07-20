import numpy as np

from ropt.ensemble_evaluator._utils import _get_failed_realizations


def test_failed_realizations01() -> None:
    failed_realizations = _get_failed_realizations(np.array([[np.nan, 1.0]]), None, 0)
    assert np.all(failed_realizations == [True])


def test_failed_realizations02() -> None:
    failed_realizations = _get_failed_realizations(
        np.array([[1.0, 1.0]]), np.array([[5 * [np.nan, 1.0]]]), 1
    )
    assert np.all(failed_realizations == [True])


def test_failed_realizations03() -> None:
    failed_realizations = _get_failed_realizations(
        np.array([[1.0, 1.0]]),
        np.array(
            [[[1.0, 1.0], [1.0, 1.0], [np.nan, 1.0], [np.nan, 1.0], [np.nan, np.nan]]]
        ),
        4,
    )
    assert np.all(failed_realizations == [True])


def test_failed_realizations04() -> None:
    failed_realizations = _get_failed_realizations(
        np.array([[1.0, 1.0]]),
        np.array(
            [[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [np.nan, 1.0]]],
        ),
        4,
    )
    assert np.all(failed_realizations == [False])


def test_failed_realizations05() -> None:
    failed_realizations = _get_failed_realizations(
        np.array([[1.0, 1.0], [1.0, 1.0]]),
        np.array([[[1.0, 1.0]], [[np.nan, 1.0]]]),
        1,
    )
    assert np.all(failed_realizations == [False, True])


def test_failed_realizations06() -> None:
    failed_realizations = _get_failed_realizations(
        np.array([[1.0, 1.0], [np.nan, 1.0]]),
        np.array([[[1.0], [1.0]], [[1.0], [1.0]]]),
        1,
    )
    assert np.all(failed_realizations == [False, True])


def test_failed_realizations07() -> None:
    failed_realizations = _get_failed_realizations(
        np.array([[1.0, 1.0], [np.nan, np.nan]]),
        np.array([[[1.0], [1.0]], [[1.0], [1.0]]]),
        1,
    )
    assert np.all(failed_realizations == [False, True])


def test_failed_realizations08() -> None:
    failed_realizations = _get_failed_realizations(
        np.array([[1.0, 1.0], [1.0, 1.0]]),
        np.array([[[1.0, 1.0], [1.0, 1.0]], [[np.nan, np.nan], [1.0, 1.0]]]),
        1,
    )
    assert np.all(failed_realizations == [False, False])
