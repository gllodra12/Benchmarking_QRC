import numpy as np
import pytest
from sympy.physics.quantum import TensorProduct

from benchmarking_qrc.hamiltonian import (
    get_coefficients,
    get_evolution_op,
    quadratic_op,
    quadratic_spin_op,
)
from benchmarking_qrc.measurements import expected_value, get_features, observables, projection_operator
from benchmarking_qrc.reservoir import cptp_map, initial_state

from utils import annihil_b, creation_b, annihil_f, creation_f, id, sigma_z, spin_plus, spin_minus


# fmt: off
@pytest.fixture
def reservoir():
    """ Density operator for a 3 bosons with hilbert dimension 3 """
    return np.array([
        [ 0.301, -0.053, -0.003, -0.051, -0.006,      0, -0.003,      0,      0,  0.256, -0.045, -0.002, -0.044, -0.005,      0, -0.003,      0,      0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-0.053,  0.076, -0.014,  0.051, -0.013,  0.002, -0.008,  0.003,      0, -0.045,  0.064, -0.012,  0.044, -0.011,  0.002, -0.007,  0.002,      0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-0.003, -0.014,  0.023, -0.008,  0.009,  0.001,  0.016,      0,      0, -0.002, -0.012,  0.02 , -0.007,  0.008,  0.001,  0.013,      0,      0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-0.051,  0.051, -0.008,  0.077, -0.014,  0.003, -0.015,  0.002,      0, -0.044,  0.044, -0.007,  0.066, -0.012,  0.002, -0.013,  0.002,      0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-0.006, -0.013,  0.009, -0.014,  0.041, -0.003,  0.011, -0.003,      0, -0.005, -0.011,  0.008, -0.012,  0.035, -0.002,  0.009, -0.003,      0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
        [     0,  0.002,  0.001,  0.003, -0.003,  0.017,      0,  0.014, -0.001,      0,  0.002,  0.001,  0.002, -0.002,  0.014,      0,  0.012, -0.001,  0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-0.003, -0.008,  0.016, -0.015,  0.011,      0,  0.023,  0.001,      0, -0.003, -0.007,  0.013, -0.013,  0.009,      0,  0.02 ,  0.001,      0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
        [     0,  0.003,      0,  0.002, -0.003,  0.014,  0.001,  0.018,      0,      0,  0.002,      0,  0.002, -0.003,  0.012,  0.001,  0.016,      0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
        [     0,      0,      0,      0,      0, -0.001,      0,      0,  0.004,      0,      0,      0,      0,      0, -0.001,      0,      0,  0.003,  0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0.256, -0.045, -0.002, -0.044, -0.005,      0, -0.003,      0,      0,  0.219, -0.038, -0.002, -0.037, -0.004,      0, -0.002,      0,      0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-0.045,  0.064, -0.012,  0.044, -0.011,  0.002, -0.007,  0.002,      0, -0.038,  0.055, -0.01 ,  0.037, -0.01 ,  0.001, -0.006,  0.002,      0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-0.002, -0.012,  0.02 , -0.007,  0.008,  0.001,  0.013,      0,      0, -0.002, -0.01 ,  0.017, -0.006,  0.007,  0.001,  0.011,      0,      0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-0.044,  0.044, -0.007,  0.066, -0.012,  0.002, -0.013,  0.002,      0, -0.037,  0.037, -0.006,  0.056, -0.01 ,  0.002, -0.011,  0.002,      0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-0.005, -0.011,  0.008, -0.012,  0.035, -0.002,  0.009, -0.003,      0, -0.004, -0.01 ,  0.007, -0.01 ,  0.03 , -0.002,  0.008, -0.002,      0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
        [     0,  0.002,  0.001,  0.002, -0.002,  0.014,      0,  0.012, -0.001,      0,  0.001,  0.001,  0.002, -0.002,  0.012,      0,  0.01 , -0.001,  0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-0.003, -0.007,  0.013, -0.013,  0.009,      0,  0.02 ,  0.001,      0, -0.002, -0.006,  0.011, -0.011,  0.008,      0,  0.017,  0.001,      0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
        [     0,  0.002,      0,  0.002, -0.003,  0.012,  0.001,  0.016,      0,      0,  0.002,      0,  0.002, -0.002,  0.01 ,  0.001,  0.013,      0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
        [     0,      0,      0,      0,      0, -0.001,      0,      0,  0.003,      0,      0,      0,      0,      0, -0.001,      0,      0,  0.003,  0, 0, 0, 0, 0, 0, 0, 0, 0],
        [     0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
        [     0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
        [     0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
        [     0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
        [     0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
        [     0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
        [     0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
        [     0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,  0, 0, 0, 0, 0, 0, 0, 0, 0],
        [     0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,  0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
# fmt: on


@pytest.mark.parametrize(
    "particle_i, level_j, dimensions",
    [(1, 0, 3), (1, 1, 3), (1, 2, 3), (2, 0, 3), (2, 1, 3), (2, 2, 3), (3, 0, 3), (3, 1, 3), (3, 2, 3)],
)
def test_projection(reservoir, particle_i, level_j, dimensions):
    n_particles = 3

    if level_j == 0:
        energy_op = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    elif level_j == 1:
        energy_op = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    elif level_j == 2:
        energy_op = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])

    if particle_i == 1:  # |j><j| x I x I
        custom_projector = np.kron(energy_op, np.kron(np.eye(dimensions), np.eye(dimensions)))
    elif particle_i == 2:  # I x |j><j| x I
        custom_projector = np.kron(np.eye(dimensions), np.kron(energy_op, np.eye(dimensions)))
    elif particle_i == 3:  # I x I x |j><j|
        custom_projector = np.kron(np.eye(dimensions), np.kron(np.eye(dimensions), energy_op))

    projector_op = projection_operator(n_particles, particle_i, level_j, dimensions)
    population = expected_value(projector_op.todense(), reservoir)
    population_sparse = expected_value(projector_op, reservoir, sparse=True)

    assert population == pytest.approx(np.trace(np.matmul(custom_projector, reservoir)), 1e-9)
    assert population_sparse == pytest.approx(np.trace(np.matmul(custom_projector, reservoir)), 1e-9)


def fermion_operators():
    """Fermion operators for a system with 3 particles"""
    # helpers
    id_product = TensorProduct(id(), id())
    sz_product = TensorProduct(sigma_z(), sigma_z())
    sz_creation = TensorProduct(sigma_z(), creation_f())
    sz_annihilation = TensorProduct(sigma_z(), annihil_f())

    # operators
    a1_dag = TensorProduct(creation_f(), id_product)
    a1 = TensorProduct(annihil_f(), id_product)
    a2_dag = TensorProduct(sz_creation, id())
    a2 = TensorProduct(sz_annihilation, id())
    a3_dag = TensorProduct(sz_product, creation_f())
    a3 = TensorProduct(sz_product, annihil_f())
    return a1_dag, a1, a2_dag, a2, a3_dag, a3


def boson_operators():
    """Boson operators for 3 particles in 3 dimension Hilbert space"""
    # helpers
    id_product = TensorProduct(id(3), id(3))
    id_creation = TensorProduct(id(3), creation_b())
    id_annihilation = TensorProduct(id(3), annihil_b())

    # operators
    a1_dag = TensorProduct(creation_b(), id_product)
    a1 = TensorProduct(annihil_b(), id_product)
    a2_dag = TensorProduct(id_creation, id(3))
    a2 = TensorProduct(id_annihilation, id(3))
    a3_dag = TensorProduct(id_product, creation_b())
    a3 = TensorProduct(id_product, annihil_b())
    return a1_dag, a1, a2_dag, a2, a3_dag, a3


def spin_operators():
    """Spin operators for 3 particles"""
    # helpers
    s1_plus = TensorProduct(spin_plus(), id(), id())
    s1_minus = TensorProduct(spin_minus(), id(), id())
    s2_plus = TensorProduct(id(), spin_plus(), id())
    s2_minus = TensorProduct(id(), spin_minus(), id())
    s3_plus = TensorProduct(id(), id(), spin_plus())
    s3_minus = TensorProduct(id(), id(), spin_minus())
    return s1_plus, s1_minus, s2_plus, s2_minus, s3_plus, s3_minus


def obs(obs_form, operators):
    a1_dag, a1, a2_dag, a2, a3_dag, a3 = operators
    if obs_form == "ii":
        return np.array([a1_dag * a1, a2_dag * a2, a3_dag * a3])

    elif obs_form == "ij":
        return np.array(
            [
                a1_dag * a1,
                a1_dag * a2 + a2_dag * a1,
                a1_dag * a3 + a3_dag * a1,
                a2_dag * a2,
                a2_dag * a3 + a3_dag * a2,
                a3_dag * a3,
            ]
        )


def precomputed_obs(operator, obs_form):
    if operator == "fermion":
        a1_dag, a1, a2_dag, a2, a3_dag, a3 = fermion_operators()
    elif operator == "boson":
        a1_dag, a1, a2_dag, a2, a3_dag, a3 = boson_operators()

    elif operator == "spin":
        # The correct name for spin operators would be:
        #       s1_plus, s1_minus, s2_plus, s2_minus, s3_plus, s3_minus
        # but I renamed in the following way to avoid duplicated the code.
        a1_dag, a1, a2_dag, a2, a3_dag, a3 = spin_operators()

    operators = a1_dag, a1, a2_dag, a2, a3_dag, a3
    return obs(obs_form, operators)


@pytest.mark.parametrize(
    "operator, dimensions, obs_form",
    [
        ("fermion", 2, "ii"),  # Fermions diag
        ("fermion", 2, "ij"),  # Fermions diag & non-diag
        ("boson", 3, "ii"),  # Bosons diag
        ("boson", 3, "ij"),  # Bosons diag & non-diag
        ("spin", 2, "ii"),  # Checks if obs spin and obs fermion are equivalent with diagonal obs
        ("spin", 2, "ij"),
    ],
)
def test_observables(operator, dimensions, obs_form):
    """Checks if the observable are computed correctly"""
    n_particles = 3
    if operator == "spin" and obs_form in ["i+i^2", "i-i^2"]:
        with pytest.raises(ValueError) as excinfo:
            observables(operator, n_particles, dimensions, obs_form)
        (msg,) = excinfo.value.args
        assert msg == f"obs_form={obs_form} is not valid for spins"

    else:
        assert (precomputed_obs(operator, obs_form) == observables(operator, n_particles, dimensions, obs_form)).all()


@pytest.mark.parametrize(
    "operator, n_particles, dimensions, obs_form, expected_length",
    [
        ("fermion", 4, 2, "ii", 4),
        ("fermion", 4, 2, "ij", 10),
        ("fermion", 5, 2, "ii", 5),
        ("fermion", 5, 2, "ij", 15),
        ("boson", 4, 2, "ii", 4),  # Fermions and bosons should have the same number of observables
        ("boson", 5, 3, "ij", 15),  # Dimension doesn't change the number of observables
        ("spin", 4, 2, "ii", 4),  # Spin and fermions should have the same number of observables
    ],
)
def test_observables_length(operator, n_particles, dimensions, obs_form, expected_length):
    """Checks if the observables generator is able to produce all the necessary observables that
    would be required for generating data"""
    assert expected_length == len(observables(operator, n_particles, dimensions, obs_form))


@pytest.mark.parametrize(
    "operator, n_particles, dimensions, obs_form, expected_shape",
    [
        ("fermion", 4, 2, "ii", 2**4),
        ("fermion", 4, 2, "ij", 2**4),  # Changing obs_form doest not change shape
        ("fermion", 5, 2, "ii", 2**5),
        ("boson", 4, 2, "ii", 2**4),
        ("boson", 4, 3, "ii", 3**4),
        ("boson", 5, 2, "ii", 2**5),
        ("boson", 5, 3, "ii", 3**5),
        ("spin", 4, 2, "ii", 2**4),
        ("spin", 5, 2, "ii", 2**5),
    ],
)
def test_observables_shape(operator, n_particles, dimensions, obs_form, expected_shape):
    """Checks if we get the expected shape for different number of particles and shape"""
    obs = observables(operator, n_particles, dimensions, obs_form)
    for ob in obs:
        shape = ob.shape
        assert shape[0] == shape[1]  # Checks if its a square matrix
        assert expected_shape == shape[0]


@pytest.mark.parametrize(
    "operator, n_particles, dimensions, obs_form",
    [
        ("fermion", 4, 2, "ij"),
        ("fermion", 5, 2, "ij"),
        ("boson", 4, 3, "ij"),
        ("boson", 5, 3, "ij"),
        ("spin", 4, 2, "ij"),
        ("spin", 5, 2, "ij"),
    ],
)
def test_observables_hermitian(operator, n_particles, dimensions, obs_form):
    """Checks if all observables are hermitian. Notice that we don't need to check ii since
    ij contains all the cases in ii"""
    obs = observables(operator, n_particles, dimensions, obs_form)
    for ob in obs:
        assert (ob == ob.conj().transpose()).all()


@pytest.mark.parametrize(
    "operator, n_particles, dimensions, obs_form",
    [
        ("fermion", 4, 2, "ij"),
        ("fermion", 4, 2, "ij"),
        ("boson", 3, 5, "ij"),
        ("spin", 2, 2, "ij"),
    ],
)
def test_get_features(operator, n_particles, dimensions, obs_form):
    """It checks if the dataset shape is correctly computed, it is expected to be (N, O+1)
    where N is the number of signals, O number of observables plus a bias term.

    In this test we've used a double for loop instead of broadcasting to double check that the dataset
    is computed correctly.
    """
    # Initial parameters
    dt = 10
    excited_state = 1
    amplitudes = get_coefficients(n_particles, coef_range=(0, 1), seed=3)
    signals = np.random.rand(100)  # Hardcoded 100 values to ignore raising error in get_features()
    if operator == "boson" or operator == "fermion":
        is_bosonic = True if operator == "boson" else False
        hamiltonian = quadratic_op(n_particles, is_bosonic, dimensions, amplitudes)
    elif operator == "spin":
        hamiltonian = quadratic_spin_op(n_particles, amplitudes)
    evolution_op = get_evolution_op(hamiltonian, dt)
    initial_reservoir = initial_state(n_particles, dimensions)

    # Generate reservoirs
    reservoirs = list(cptp_map(signals, initial_reservoir, evolution_op, dimensions, excited_state))

    # Generating dataset
    obs = np.array(observables(operator, n_particles, dimensions, obs_form))
    fake_dataset = np.ones([len(signals), len(obs)])
    for i, reservoir in enumerate(reservoirs):
        for j, observable in enumerate(obs):
            exp_value = np.trace(np.matmul(observable, reservoir))
            if exp_value.imag > 1e-9:
                raise ValueError(
                    "Either your density_op or your operator in expected_value() is non-hermitian and should be hermitian"
                )
            fake_dataset[i, j] = exp_value.real

    expected_dataset = fake_dataset.reshape(len(signals), len(obs))
    bias = np.ones((len(signals), 1))
    expected_dataset = np.append(expected_dataset, bias, axis=1)

    dataset, _ = get_features(reservoirs, obs)
    assert expected_dataset == pytest.approx(dataset, 1e-9)
