from collections import deque

import numpy as np
import pytest
from scipy.linalg import lstsq

from benchmarking_qrc.hamiltonian import get_coefficients
from benchmarking_qrc.run_memory_capacity import get_inputs, get_targets, main, memory_capacity


def test_lstsq():
    """Check if lstsq gets the result I computed to solve this system of equations:

            x +  y = 1
            x + 2y = 3
            x + 3y = 3
            x + 4y = 5

    As we can see this system is overdetermined, so there isn't an exact solution. However,
    we can find a x, y value that are as close as possible to satisfy all 4 equations, using
    the ordinary least square algorithm.

    scipy.lstsq computes:
        - optimal weights.
        - l2-norm of the residual ||y - opt_weights*x||^2
        - rank of the matrix to check if there is colinearity.
        - singular values of the matrix
    """
    matrix = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
    y = np.array([1, 3, 3, 5])
    computed_weights = np.array([0, 6 / 5])
    weights, res, rank, s = lstsq(matrix, y)

    computed_residual = np.linalg.norm(y - np.matmul(matrix, weights)) ** 2
    computed_rank = np.linalg.matrix_rank(matrix)
    singular_matrix = np.matmul(matrix.transpose(), matrix)
    computed_singular_values = np.sqrt(np.linalg.eig(singular_matrix)[0])

    assert computed_weights == pytest.approx(weights, 1e-3)
    assert computed_residual == pytest.approx(res, 1e-3)
    assert computed_rank == rank
    assert computed_singular_values == pytest.approx(np.flip(s), 1e-3)


def test_deque():
    """Checks if deque pop only return the last value in a generator"""
    my_generator = (x * x for x in range(3))
    result = deque(my_generator, maxlen=1).pop()
    assert 4 == result


@pytest.fixture
def signals():
    return np.array([0.1, 0.3, 0.2, 0.44, 0.5, 0.8, 0.61, 0.73, 0.92, 0.999])


@pytest.mark.parametrize(
    "predictions, targets, result",
    [
        (np.array([0.7, 0.5, 0.3]), np.array([0.2, 0.6, 0.9]), 0.993),
        (np.array([0.4, 0.5, 0.6]), np.array([0.4, 0.5, 0.6]), 1.000),
    ],
)
def test_memory_capacity(predictions, targets, result):
    sigma_p = np.std(predictions)
    sigma_t = np.std(targets)
    sigma_pt = np.mean(predictions * targets) - np.mean(predictions) * np.mean(targets)
    correlation = sigma_pt / (sigma_p * sigma_t)

    mc = memory_capacity(predictions, targets)
    assert correlation**2 == pytest.approx(mc, 1e-6)
    assert result == pytest.approx(mc, 1e-3)


@pytest.mark.parametrize(
    "delay, result",
    [
        (1, np.array([0.3, 0.2, 0.44, 0.5, 0.8, 0.61, 0.73, 0.92, 0.999])),
        (2, np.array([0.2, 0.44, 0.5, 0.8, 0.61, 0.73, 0.92, 0.999])),
        (4, np.array([0.5, 0.8, 0.61, 0.73, 0.92, 0.999])),
    ],
)
def test_get_inputs(signals, delay, result):
    assert (result == get_inputs(signals, delay)).all()


@pytest.mark.parametrize(
    "delay, degree, result",
    [
        (1, 1, np.array([0.1, 0.3, 0.2, 0.44, 0.5, 0.8, 0.61, 0.73, 0.92])),
        (2, 3, np.array([0.1, 0.3, 0.2, 0.44, 0.5, 0.8, 0.61, 0.73]) ** 3),
    ],
)
def test_get_targets(signals, delay, degree, result):
    assert (result == get_targets(signals, delay, degree)).all()


@pytest.mark.parametrize(
    "dimensions, operator, degree, exp_train_mc, exp_test_mc",
    [
        (3, "boson", 1, 0.6200565884230689, 0.40656561338424463),
        (2, "fermion", 1, 0.7788241921482835, 0.025067771760359077),
        (2, "spin", 1, 0.5151372945428282, 0.22317823433905123),
    ],
)
def test_main_memory_capacity(dimensions, operator, degree, exp_train_mc, exp_test_mc):
    "Test to check consistency for different versions of memory_capacity.py"
    n_particles = 4
    seed = 3
    coefficients = get_coefficients(
        n_particles, coef_onsites=[1, 1, 1, 1], coef_couplings=[0.5, 0.4, 0.3, 0.5, 0.4, 0.5], seed=seed
    )
    delay = 4
    obs_form = "ij"
    wash_time, train_time, test_time = 100, 20, 10
    excited_state = 1
    dt = 10

    _, _, train_mc, test_mc = main(
        n_particles,
        dimensions,
        coefficients,
        operator,
        delay,
        obs_form,
        wash_time,
        train_time,
        test_time,
        excited_state,
        dt,
        degree=degree,
    )

    assert exp_train_mc == pytest.approx(train_mc, 1e-9)
    assert exp_test_mc == pytest.approx(test_mc, 1e-9)
