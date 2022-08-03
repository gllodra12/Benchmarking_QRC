from itertools import combinations_with_replacement

import numpy as np
from openfermion.linalg.sparse_tools import boson_operator_sparse, get_sparse_operator
from openfermion.ops import BosonOperator, FermionOperator
from scipy import sparse

from benchmarking_qrc.hamiltonian import spin_minus, spin_plus


def expected_value(operator, density_op, sparse=False):
    """Computes either the expected value (using an observable) or the probability
    (using a projector op) after a measurement using the following equation:

        expected value = Tr(Operator * density_op)

    Args:
        operator (np.array): Use the matrix defined in the following functions: observable(), projection().
        density_op (np.array): Matrix describing the state of the system your modeling.

    Returns:
        float: Expected value or probability after a measurement.
    """
    if sparse:
        return operator.dot(density_op).diagonal().sum()
    return np.einsum("ij,ji->", operator, density_op)


def observables(operator, n_particles, dimensions=2, obs_form="ij"):
    """Generates an array with all the possible observables for a specific `obs_form`.

    It generates the diagonal observables (ii) or the product of observables (ij).

    Args:
        operator (str): Type of particles in the reservoir (boson, fermion, spin)
        n_particles (int): Number of particles in the system.
        dimensions (int, optional): Number of particles on the same site (dim_boson=n, dim_fermion=2).
        obs_form (str, optional): Choose an observable to measure the reservoir.
            There are two observables "ii" (a^_i a_i) and "ij" (a^_i a_j). Defaults to "ij".

    Raises:
        NotImplementedError: Currently only obs_form "ii" and "ij" are implemented.

    Returns:
        np.array: Array with all possible observables of the form `obs_form`.
    """
    # Define particle algebra
    if operator == "fermion":
        particle = FermionOperator
    elif operator == "boson":
        particle = BosonOperator
    elif operator == "spin":
        particle = "spin"

    all_observables = list()

    # Choose which observable form we want to compute
    if obs_form == "ii":
        observable_generator = diag_observables(n_particles, particle)
    elif obs_form == "ij":
        observable_generator = product_observables(n_particles, particle)
    else:
        raise NotImplementedError("The observable form {} has yet not been implemented".format(obs_form))

    for op, observable_idx in observable_generator:
        # Get observable in sparse matrix form
        if particle == FermionOperator:
            observable = get_sparse_operator(op)
        elif particle == BosonOperator:
            observable = boson_operator_sparse(op, dimensions)
        elif particle == "spin":
            observable = get_sparse_operator(op)

        # Get the final observable. If our observable acts on particle 1 & 2 but the system has
        # 3 particles we need to add the identity to act on particle 3
        while observable_idx < n_particles:
            observable = sparse.kron(observable, sparse.identity(dimensions))
            observable_idx += 1

        if observable.imag.sum() > 1e-9:
            raise ValueError("Unexpected observable operator with imaginary values")

        all_observables.append(observable.todense())
    return np.array(all_observables)


def diag_observables(n_particles, particle):
    """Generate all diagonal observables of the form a^_i a_i.

    For n=3, we expect these 3 observables: a^_0 a_0, a^_1 a_1, a^_2 a_2

    Args:
        n_particles (int): Number of particles in the system
        particle (BosonOperator or FermionOperator or "spin"): Determines if the particle is a boson, a fermion or a spin.

    Yields:
        BosonOperator, FermionOperator or QubitOperator: Diagonal observables
        int: For a system of n particles it tells onto which of the n particle the observable acts on.
    """

    for particle_idx in range(n_particles):
        if particle == FermionOperator or particle == BosonOperator:
            observable_op = particle(((particle_idx, 1), (particle_idx, 0)))
        elif particle == "spin":
            observable_op = spin_plus(particle_idx) * spin_minus(particle_idx)

        # Python indexing starts from 0 but I label the particle from 1 to n.
        observable_idx = particle_idx + 1
        yield observable_op, observable_idx


def product_observables(n_particles, particle):
    """Generate all the pair of two observables (diagonal and non-diagonal)

    For n=3, we expect these 6 observables:
        a^_0 a_0
        a^_1 a_0 + a^_0 a_1
        a^_2 a_0 + a^_0 a_2
        a^_1 a_1
        a^_1 a_2 + a^2 a_1
        a^_2 a_2

    Args:
        n_particles (int): Number of particles in the system
        particle (BosonOperator or FermionOperator or "spin"): Determines if the particle is a boson, a fermion or a spin.

    Yields:
        BosonOperator or FermionOperator or QubitOperator: Diagonal and non-diagonal observables
        int: For a system of n particles it tells onto which of the n particle the observable acts on.
    """
    for source, target in combinations_with_replacement(range(n_particles), 2):
        if source == target:
            # operator form: a^_i a_i
            if particle == FermionOperator or particle == BosonOperator:
                observable_op = particle(((source, 1), (target, 0)))
            elif particle == "spin":
                observable_op = spin_plus(source) * spin_minus(target)
            observable_idx = source + 1
        else:
            # operator form: a^_i a_j + a^_j a_i
            if particle == FermionOperator or particle == BosonOperator:
                observable_op = particle(((source, 1), (target, 0))) + particle(((target, 1), (source, 0)))
            elif particle == "spin":
                observable_op = spin_plus(source) * spin_minus(target) + spin_plus(target) * spin_minus(source)
            observable_idx = max(source, target) + 1
        yield observable_op, observable_idx


def projection_operator(n_particles, particle_i, level_j, dimensions):
    """Computes the projector operator, then you can use the function expected_value
    and get the probability to find particle_i on level_j.

    Args:
        n_particles (int): Number of particles in the system.
        particle_i (int): Select a particle from 1 to n to see its population.
        level_j (int): Select an energy level from 0 to dimensions to see its population.
        dimensions (int): Bosons have a dimensions = n (where n goes from 0 to inf), while fermions have hilbert dimensions = 2.

    Returns:
        sparse.csc_matrix: An operator to measure the population on a specific particle (particle_i) and energy level (level_j).
    """
    # Define energy level |j> and the matrix |j><j|
    energy_lvl = np.zeros((int(dimensions), 1))
    energy_lvl[level_j, 0] = 1
    energy_op = sparse.csc_matrix(energy_lvl * energy_lvl.T)

    # Projector opeator onto the first particle --> |j><j| x I x I ...
    if particle_i == 1:
        projector_op = sparse.kron(energy_op, sparse.identity(dimensions ** (n_particles - 1)))

    # Projector operator onto the last particle --> I x I ... x |j><j|
    elif n_particles == particle_i:
        projector_op = sparse.kron(sparse.identity(dimensions ** (particle_i - 1)), energy_op)

    # Projector operator onto particle i --> I x ... |j><j| ... x I
    else:
        projector_op = sparse.kron(sparse.identity(dimensions ** (particle_i - 1)), energy_op)
        projector_op = sparse.kron(projector_op, sparse.identity(dimensions ** (n_particles - particle_i)))

    return projector_op


def frobenius_distance(density_op1, density_op2):
    matrix = density_op1 - density_op2
    return np.linalg.norm(matrix, "fro")


def get_features(reservoirs, observables, save_iterations=list(range(1200, 1211))):
    """Generates a dataset to train the ML algorithm to remember past inputs. with shape (L, n+1):

    The dataset is a matrix with shape (L, n+1) where n, L are the number of observables and
    reservoirs respectively. Also notice that we add a bias term 1 in the last column.

        [[Tr(O_1 ρ_1) Tr(O_2 ρ_1) ... Tr(O_n ρ_1) 1],
         [Tr(O_1 ρ_2) ...             Tr(O_n ρ_2) 1],
         [...                                      ],
         [Tr(O_1 ρ_L) ...             Tr(O_n ρ_L) 1]]

    Args:
        reservoirs (generator): Generates a new reservoir from the previous reservoir state.
        observables (np.array): Contains the different observable to measure the reservoir

    Returns:
        np.array: Dataset use to make predictions
        np.array: Save several snapshots of the reservoir during the evolution.
    """
    data = list()
    dm_snapshots = list()
    for iteration, reservoir in enumerate(reservoirs):
        row = np.einsum("kij,ji->k", observables, reservoir)

        if any(row.imag > 1e-9):
            raise ValueError("Either some observable or the reservoir are non-hermitian")
        data.append(row.real)

        # Take several snapshots of the system after the training process
        if iteration in save_iterations:
            dm_snapshots.append(reservoir)

    data = np.array(data)

    # Checks a necessary condition for OLS algorithm
    num_reservoirs, num_observables = data.shape
    if num_reservoirs <= num_observables + 1:
        raise ValueError(
            "The dataset is not overdetermined, train time ({}) must be higher that the number of observables ({})".format(
                num_reservoirs, num_observables + 1
            )
        )

    bias = np.ones((num_reservoirs, 1))
    return np.append(data, bias, axis=1), np.array(dm_snapshots)
