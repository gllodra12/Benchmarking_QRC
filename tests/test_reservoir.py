import numpy as np
import pytest

from benchmarking_qrc.reservoir import evolve, get_input_state, insert_input


@pytest.fixture
def signals():
    return np.array([0.1, 0.4])


@pytest.fixture
def simple_input():
    return np.array([np.sqrt(0.5), np.sqrt(0.5)]).reshape(-1, 1)


@pytest.fixture
def simple_reservoir():
    return np.array([[0.25, 0, 0.43, 0], [0, 0, 0, 0], [0.43, 0, 0.75, 0], [0, 0, 0, 0]])


@pytest.fixture()
def evolution_op():
    return np.array(
        [
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.39247597 + 0.29739973j, -0.72959031 + 0.47456716j, 0.0 + 0.0j],
            [0.0 + 0.0j, -0.72959031 + 0.47456716j, 0.11276125 + 0.47934214j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, -0.40538818 + 0.91414464j],
        ]
    )


@pytest.mark.parametrize(
    "dimensions, excited_state, signal, result",
    [
        (2, 1, 0.25, np.array([np.sqrt(0.25), np.sqrt(1 - 0.25)])),
        (3, 2, 0.25, np.array([np.sqrt(0.25), 0, np.sqrt(1 - 0.25)])),
        (5, 1, 0.25, np.array([np.sqrt(0.25), np.sqrt(1 - 0.25), 0, 0, 0])),
        (5, 2, 0.25, np.array([np.sqrt(0.25), 0, np.sqrt(1 - 0.25), 0, 0])),
    ],
)
def test_get_input_state(dimensions, excited_state, signal, result):
    """Checks if input state is compute correctly and if it is properly normalized."""
    state = get_input_state(signal, dimensions, excited_state)
    inner_product = np.dot(state.conj().transpose(), state)
    assert state == pytest.approx(result.reshape(dimensions, -1), 1e-6)
    assert pytest.approx(inner_product, 1e-9) == 1


def test_insert_input(simple_input, simple_reservoir):
    # rho = |Ψ><Ψ| being |Ψ> the input state
    rho = np.array([[0.5, 0.5], [0.5, 0.5]])

    # From simple reservoir if trace out idx 0 and keep idx 1 we get:
    reduce_dm = np.array([[1, 0], [0, 0]])

    # Final reservoir is the tensor product between the rho (input) and the
    # reduce_dm (reservoir without input)
    reservoir = np.kron(rho, reduce_dm)
    assert (np.around(insert_input(simple_input, simple_reservoir), 1) == reservoir).all()


@pytest.mark.parametrize(
    "n_particles, dimensions, result",
    [
        (3, 2, 2**3),  # 3 fermions with 2 levels
        (3, 3, 3**3),  # 3 bosons with 3 levels
        (2, 4, 4**2),  # 2 bosons with 4 levels
        (5, 3, 3**5),
    ],
)
def test_shape_insert_input(n_particles, dimensions, result):
    """Checks if insert_input outputs the right reservoir shape
    for different number of particles and dimensions.
    """
    signal = np.random.uniform(low=0, high=1)
    input_state = get_input_state(signal, dimensions, dimensions - 1)
    shape = dimensions**n_particles
    reservoir = np.random.rand(shape, shape)
    assert insert_input(input_state, reservoir).shape == (result, result)


def test_simple_evolve(simple_reservoir, evolution_op):
    """Simply checks if we are computing the following formula correctly
    ρ_new = e^{-iHt} * ρ * e^{iHt}
    """
    e_neg_iHt = evolution_op
    e_iHt = np.conj(evolution_op)
    reservoir = np.dot(simple_reservoir, e_iHt)  # Even if we transpose the result doesn't change.
    reservoir = np.dot(e_neg_iHt, reservoir)

    expected_reservoir = evolve(simple_reservoir, evolution_op)
    assert reservoir == pytest.approx(expected_reservoir, 1e-6)
