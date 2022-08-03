import numpy as np
from openfermion.linalg.sparse_tools import get_sparse_operator
import pytest
from scipy import sparse

from benchmarking_qrc.hamiltonian import (
    get_coefficients,
    get_evolution_op,
    quadratic_op,
    quadratic_spin_op,
    spin_plus,
    spin_minus,
)
from utils import bosonic_op, bosonic_op_3, fermionic_op, fermionic_op_3, spin_op, spin_op_3


@pytest.fixture
def fermionic_hamiltonian():
    return sparse.csc_matrix([[0, 0, 0, 0], [0, 0.5, 0.4, 0], [0, 0.4, 0.5, 0], [0, 0, 0, 1]])


def custom_quadratic(n_particles, coefficients, is_bosonic):
    """We choose the same random seed, as in hamiltonian.py to test if the code works as expected.

    Args:
        coefficients (tuple(list, list)): Coefficients for the onsite elements (J_ii) and coupling elements (J_ij)
        op (dict): Either bosonic_op or fermionic_op are the expected dicts

    Returns:
        sympy.matrices: Hamiltonian for a bosonic/fermionic system of 2 particles.
    """
    if is_bosonic:
        op = bosonic_op if n_particles == 2 else bosonic_op_3
    else:
        op = fermionic_op if n_particles == 2 else fermionic_op_3

    if n_particles == 2:
        (J_11, J_22), (J_12,) = coefficients
        return J_11 * op["a1^dag a1"] + J_12 * (op["a1^dag a2"] + op["a2^dag a1"]) + J_22 * op["a2^dag a2"]

    elif n_particles == 3:
        (J_11, J_22, J_33), (J_12, J_13, J_23) = coefficients
        return (
            J_11 * op["a1^dag a1"]
            + J_12 * (op["a1^dag a2"] + op["a2^dag a1"])
            + J_13 * (op["a1^dag a3"] + op["a3^dag a1"])
            + J_22 * op["a2^dag a2"]
            + J_23 * (op["a2^dag a3"] + op["a3^dag a2"])
            + J_33 * op["a3^dag a3"]
        )


def custom_transverse(coefficients, op, h):
    (J_11, J_12), (J_21, J_22) = coefficients
    return (
        J_11 * op["a1^dag a1"]
        + J_12 * op["a1^dag a2"]
        + J_21 * op["a2^dag a1"]
        + J_22 * op["a2^dag a2"]
        + h * (op["a1^dag a1"] + op["a2^dag a2"])
    )


@pytest.mark.parametrize("n_particles", [(2), (3)])
def test_hamiltonian_fermion(n_particles):
    """Testing the quadratic hamiltonian for a fermionic system
    of 2 particles.
    """
    coefficients = get_coefficients(n_particles, coef_range=(0, 1), seed=3)
    result = quadratic_op(n_particles, is_bosonic=False, dimensions=2, coefficients=coefficients)
    expected = np.array(custom_quadratic(n_particles, coefficients, is_bosonic=False)).astype(complex)
    assert result.toarray() == pytest.approx(expected, 1e-3)


@pytest.mark.parametrize("n_particles", [(2), (3)])
def test_hamiltonian_boson(n_particles):
    """Testing the quadratic hamiltonian for a boson system
    of 2 particles and 3 level occupation number.
    """
    coefficients = get_coefficients(n_particles, coef_range=(0, 1), seed=3)
    result = quadratic_op(n_particles, is_bosonic=True, dimensions=3, coefficients=coefficients)
    expected = np.array(custom_quadratic(n_particles, coefficients, is_bosonic=True)).astype(complex)
    assert result.toarray() == pytest.approx(expected, 1e-3)


@pytest.mark.parametrize("dt", [(1), (5), (10)])
def test_get_evolution_op(fermionic_hamiltonian, dt):
    """Checks if the get_evolution_op computes the following formula correctly.
        e^{-iHΔt} = V e^{-iDΔt} V^{-1}

    To compute e^{-iHΔt} we need to follow these steps.
        1. Compute eigenvalues and eigenvectors of H.
        2. Write all eigenvalues to the power of e in a diagonal matrix.
        3. Compute the matrix product  V e^{-iDΔt} V^{-1} where V is the matrix of eigenvectors
    """
    # 1st Step: I compute the eigenvalues and eigenvectors.
    eigvalues = [0, 0.1, 0.9, 1]
    eigvectors = np.array(
        [[1, 0, 0, 0], [0, -1 / np.sqrt(2), 1 / np.sqrt(2), 0], [0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0], [0, 0, 0, 1]]
    )

    # 2nd Step: Writing e^{-iDΔt} where D are all the eigenvalues
    exp_diag = np.zeros((4, 4), dtype=np.complex64)
    exp_diag[0, 0] = np.exp(-1j * eigvalues[0] * dt)
    exp_diag[1, 1] = np.exp(-1j * eigvalues[1] * dt)
    exp_diag[2, 2] = np.exp(-1j * eigvalues[2] * dt)
    exp_diag[3, 3] = np.exp(-1j * eigvalues[3] * dt)

    # 3rd Steo: Computing the matrix product V e^{-iDΔt} V^{-1}
    inv_V = np.linalg.inv(eigvectors)
    evolution_op = np.dot(eigvectors, exp_diag)
    evolution_op = np.dot(evolution_op, inv_V)

    assert evolution_op == pytest.approx(get_evolution_op(fermionic_hamiltonian, dt), 1e-6)


def test_spin_ops():
    assert (get_sparse_operator(spin_plus(0)).todense() == np.array([[0, 1], [0, 0]])).all()
    assert (get_sparse_operator(spin_minus(0)).todense() == np.array([[0, 0], [1, 0]])).all()


@pytest.mark.parametrize("n_particles", [(2), (3)])
def test_spin_hamiltonian(n_particles):
    coefficients = get_coefficients(n_particles, coef_range=(0, 1), seed=3)
    if n_particles == 2:
        (J_11, J_22), (J_12,) = coefficients
        expected = (
            J_11 * spin_op["s1+ s1-"] + J_12 * (spin_op["s1+ s2-"] + spin_op["s2+ s1-"]) + J_22 * spin_op["s2+ s2-"]
        )
    elif n_particles == 3:
        (J_11, J_22, J_33), (J_12, J_13, J_23) = coefficients
        expected = (
            J_11 * spin_op_3["s1+ s1-"]
            + J_12 * (spin_op_3["s1+ s2-"] + spin_op_3["s2+ s1-"])
            + J_13 * (spin_op_3["s1+ s3-"] + spin_op_3["s3+ s1-"])
            + J_22 * spin_op_3["s2+ s2-"]
            + J_23 * (spin_op_3["s2+ s3-"] + spin_op_3["s3+ s2-"])
            + J_33 * spin_op_3["s3+ s3-"]
        )
    result = quadratic_spin_op(n_particles, coefficients)
    assert result.toarray() == pytest.approx(np.array(expected).astype(complex), 1e-3)


@pytest.mark.parametrize(
    "n_particles, coef_range, coef_onsites, coef_couplings, seed, expected",
    [
        (4, [0, 1], None, None, 3, (np.array([0.55, 0.71, 0.29, 0.51]), np.array([0.89, 0.9, 0.13, 0.21, 0.05, 0.44]))),
        (3, None, [0.8, 0.7, 0.6], [0.2, 0.3, 0.4], None, ([0.8, 0.7, 0.6], [0.2, 0.3, 0.4])),
        (4, None, [1, 1, 1, 1], [2, 2, 2, 2, 2, 2], None, ([1, 1, 1, 1], [2, 2, 2, 2, 2, 2])),
    ],
)
def test_get_coefficients(n_particles, coef_range, coef_onsites, coef_couplings, seed, expected):
    coefficients = get_coefficients(n_particles, coef_range, coef_onsites, coef_couplings, seed)

    if coef_range:
        assert type(coefficients) == type(expected)
        assert len(coefficients) == len(expected)
        assert np.round(coefficients[0], 2) == pytest.approx(expected[0], 1e-2)
        assert np.round(coefficients[1], 2) == pytest.approx(expected[1], 1e-2)
    else:
        assert coefficients == expected


@pytest.mark.parametrize(
    "n_particles, coef_range, coef_onsites, coef_couplings, seed, exp_msg",
    [
        (
            4,
            [0, 1, 2],
            None,
            None,
            3,
            "Amp range must contain only 2 values. First value must be the lowest and the second value the highest to be generated.",
        ),
        (
            3,
            None,
            [0.8, 0.7, 0.6, 1],
            [0.2, 0.3, 0.4],
            None,
            "The length of len(coef_onsite)=4. The expected length is 3",
        ),
        (3, None, [1, 1, 1], [2, 2], None, "The length of len(coef_coupling)=2. The expected length is 3"),
    ],
)
def test_get_coefficients_errors(n_particles, coef_range, coef_onsites, coef_couplings, seed, exp_msg):
    with pytest.raises(ValueError) as excinfo:
        get_coefficients(n_particles, coef_range, coef_onsites, coef_couplings, seed)
    (msg,) = excinfo.value.args
    assert msg == exp_msg
