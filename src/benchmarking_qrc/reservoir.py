import logging

from cirq import partial_trace
import numpy as np

from benchmarking_qrc.utils import trace_equal_one


logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(asctime)s:%(name)s:%(lineno)d:%(message)s")

filename = "reservoir_errors.log"
file_handler = logging.FileHandler(filename)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def initial_state(n_particles, dimensions):
    """The initial state of the system is of the form

        |0><0| x |0><0| x ... x |0><0|

    where <0|=np.array([1, 0]).

    Args:
        n_particles (int): Number of particles in the system
        dimensions (int): Number of particles on the same site (dim_boson=n, dim_fermion=2).

    Returns:
        array : Density matrix full of zeros except at the position (0, 0) with 1.
    """
    N = pow(dimensions, n_particles)
    matrix = np.zeros((N, N))
    matrix[0, 0] = 1
    return matrix


def basis(dimensions, excited_state):
    """Defines the computational basis for different particles.

    For fermions the computational basis is |0> or |1>.
    For bosons the computational basis is |0>, |1>, ..., |n>.

    Args:
        dimensions (int): Number of particles on the same site (dim_boson=n, dim_fermion=2).
        excited_state (int): Defines the energy level of the excited state (excited_state ∊ (1, n)).

    Raises:
        ValueError: excited_state can't be larger than dimensions.

    Returns:
        np.array: Quantum state of particle
    """
    if excited_state >= dimensions:
        raise ValueError(f"Excited level is equal or larger than dimensions. Make sure excited_state < dimensions.")
    state = np.zeros(dimensions).reshape(dimensions, -1)
    state[excited_state] = 1
    return state


def get_input_state(signal, dimensions, excited_state):
    """Given an input data (signal) it return a quantum state that contains
    the classical data.

        Example:
            - If we excite level 2 (excited_state=2) with signal=x, dimensions=4, it returns:
                √x|0> + √(1-x)|2> = np.array([[√x], [0], [√(1-x)], [0]])

    Args:
        signal (float or np.array): Number that is feed into the quantum system.
        dimensions (int): Number of energy levels in the quantum system.
        excited_state (int): Defines the energy level of the excited state.

    Returns:
        np.array: Quantum state containing the input data
    """
    return np.sqrt(signal) * basis(dimensions, 0) + np.sqrt(1 - signal) * basis(dimensions, excited_state)


def insert_input(input_state, reservoir):
    """Introduce an input state into the first particle (in python the first
    particle is the particle 0) of the reservoir.

    Args:
        input_state (np.array): Superposition state of the form P_0|0> + P_n|n>.
        reservoir (np.array): Matrix describing the state of the system your modeling

    Returns:
        np.array: Reservoir after inserting the input.
    """
    dimensions = len(input_state)
    n_particles = round(np.log(reservoir.shape[0]) / np.log(dimensions))
    rho = np.outer(input_state, input_state)

    # Trace out the particle at index 0, so we keep all particles from 1 until n_particles-1
    reduce_dm = partial_trace(reservoir.reshape([dimensions] * 2 * n_particles), range(1, n_particles))

    # Reshape to a 2D-matrix
    newshape = dimensions ** (n_particles - 1)
    reduce_dm = reduce_dm.reshape(newshape, newshape)
    return np.kron(rho, reduce_dm)


def evolve(density_op, evolution_op, raising_error=True):
    """Let the density operator (ρ) evolve following the dynamics given
     by the hamiltonian H for a period of time Δt.

         ρ(Δt) = e^{-iHΔt} * ρ * e^{iHΔt}

     Notice that e^{-iHΔt} is the evolution_op computed in the function get_evolution_op()

     Args:
         density_op (np.array): Matrix describing the state of the system your modeling.
         evolution_op (np.array): Unitary evolution of the system for a period of time dt

    Returns:
         (np.array): Density operator after evolving for a period of time dt.
    """
    if raising_error:
        if not trace_equal_one(density_op):
            logger.error("Incorrect density operator:")
            logger.error(density_op)
            logger.error("Because it's trace is: {}".format(np.trace(density_op)))
            raise ValueError("Density operator doesn't have trace = 1 (see {} for more info)".format(filename))

    return np.matmul(evolution_op, np.matmul(density_op, evolution_op.transpose().conj()))


def cptp_map(signals, initial_reservoir, evolution_op, dimensions, excited_state):
    """At each iteration we input a signal into the reservoir and let the system evolve
    through its natural dynamics. After each iteration we take an snapshot of the reservoir
    and store all the snapshots in a list.

    Args:
        signals (np.array): Sequence of random numbers, once this sequence have finished we stop iterating
        initial_reservoir (np.array): Matrix representing the state of the reservoir at time 0
        evolution_op (np.array): Unitary evolution of the system for a period of time dt
        dimensions (int): Number of particles on the same site (dim_boson=n, dim_fermion=2).
        excited_state (int): The input into the reservoir is linear combination of the ground state (|0>) an excited state (|n>)

    Yields:
        np.array: Reservoir state after inserting input and evolving
    """
    # Both reservoirs rho_a and rho_b must have the same inputs.
    new_reservoir = initial_reservoir
    for signal in signals:
        input_state = get_input_state(signal, dimensions, excited_state)

        new_reservoir = insert_input(input_state, new_reservoir)
        new_reservoir = evolve(new_reservoir, evolution_op, raising_error=True)

        yield new_reservoir
