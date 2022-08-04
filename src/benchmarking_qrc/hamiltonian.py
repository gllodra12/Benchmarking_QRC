from itertools import combinations

import numpy as np
from openfermion.linalg.sparse_tools import boson_operator_sparse, get_sparse_operator
from openfermion.ops.operators import BosonOperator, FermionOperator, QubitOperator
from scipy.linalg import expm

from benchmarking_qrc.utils import is_unitary


def quadratic_op(n_particles, is_bosonic, dimensions, coefficients, raising_error=True):
    """Hamiltonian of the form:

    H=\sum_{i,j}^{N} J_{ij} a_{i}^\dagger a_{j}

    Args:
        n_particles (int): Number of particles in the system.
        is_bosonic (bool): Particles are either bosons (True) or fermions (False).
        dimensions (int): Number of particles on the same site (dim_boson=n, dim_fermion=2).
        coefficients (tuple(list, list)): Coefficients for the onsite elements (J_ii) and coupling elements (J_ij)
        raising_error (bool, optional): Checks if the hamiltonian is hermitian. Defaults to True.

    Returns:
        sparse.csc: Matrix describing how the system will evolve.
    """
    if is_bosonic == False and dimensions != 2:
        raise ValueError(f"Fermions have dimensions=2, not dimensions={dimensions}")

    # Initialize the hamiltonian
    particle = BosonOperator if is_bosonic else FermionOperator
    coef_onsites, coef_couplings = coefficients
    onsite_hamiltonian, coupling_hamiltonian = particle(), particle()

    # Add on-site energy
    for particle_i in range(n_particles):
        onsite_hamiltonian += particle(((particle_i, 1), (particle_i, 0)), coef_onsites[particle_i])

    # Add coupling energy
    for idx, (source, target) in enumerate(combinations(range(n_particles), 2)):
        coupling_hamiltonian += particle(((source, 1), (target, 0)), coef_couplings[idx]) + particle(
            ((target, 1), (source, 0)), coef_couplings[idx]
        )

    # Transform hamiltonian into sparse matrix
    onsite_hamiltonian = ham_to_matrix(onsite_hamiltonian, particle, dimensions, raising_error)
    coupling_hamiltonian = ham_to_matrix(coupling_hamiltonian, particle, dimensions, raising_error)

    return onsite_hamiltonian + coupling_hamiltonian


def quadratic_spin_op(n_particles, coefficients, raising_error=True):
    """Hamiltonian with spin ladder operator S+ and S-

    H=\sum_{i,j}^{N} J_{ij} S+_{i} S-_{j}

    Args:
        n_particles (int): Number of particles in the system.
        coefficients (tuple(list, list)): Coefficients for the onsite elements (J_ii) and coupling elements (J_ij)
        raising_error (bool, optional): Checks if the hamiltonian is hermitian. Defaults to True.

    Raises:
        ValueError: Hamiltonian must be hermitian

    Returns:
        sparse.csc: Matrix describing how the system will evolve.
    """
    # Initialize the hamiltonian
    coef_onsite, coef_couplings = coefficients
    onsite_hamiltonian, coupling_hamiltonian = 0, 0

    # Add on-site energy
    for particle_i in range(n_particles):
        onsite_hamiltonian += coef_onsite[particle_i] * spin_plus(particle_i) * spin_minus(particle_i)

    # Add coupling energy
    for idx, (source, target) in enumerate(combinations(range(n_particles), 2)):
        coupling_hamiltonian += coef_couplings[idx] * (
            spin_plus(source) * spin_minus(target) + spin_plus(target) * spin_minus(source)
        )

    # Transform hamiltonian into sparse matrix
    onsite_hamiltonian = get_sparse_operator(onsite_hamiltonian)
    coupling_hamiltonian = get_sparse_operator(coupling_hamiltonian)

    hamiltonian = onsite_hamiltonian + coupling_hamiltonian
    if raising_error:
        if not is_hermitian(hamiltonian.todense()):
            raise ValueError("The hamiltonian is not hermitian")

    return hamiltonian


def ham_to_matrix(hamiltonian, particle, dimensions, raising_error):
    """Transform a symbolic representation of a hamiltonian into a matrix.

    Args:
        hamiltonian (BosonOperator or FermionOperator): Hamiltonian in symbolic form using openfermion operators.
        particle (BosonOperator or FermionOperator): Particle is a boson or a fermion
        dimensions (int, optional): Number of particles on the same site (dim_boson=n, dim_fermion=2).
        raising_error (bool, optional): Check if the hamiltonian is hermitian.

    Returns:
        sparse.csc: hamiltonian in matrix form
    """
    if particle == FermionOperator:
        hamiltonian = get_sparse_operator(hamiltonian)
    elif particle == BosonOperator:
        if dimensions == None:
            raise ValueError("An upper-bound for boson occupation level is expected")
        hamiltonian = boson_operator_sparse(hamiltonian, dimensions)

    if raising_error:
        if not is_hermitian(hamiltonian.todense()):
            raise ValueError("The hamiltonian is not hermitian")

    return hamiltonian


def get_coefficients(n_particles, coef_range=None, coef_onsites=None, coef_couplings=None, seed=None):
    """Generates an array with the onsite coefficients J_ii and the coupling coefficients J_ij.
    This seed will also generate the sequence of signal values that will be injected into the reservoir.

    Args:
        n_particles (int): Number of particles in the system
        coef_range (list):  Randomly generate all coefficients with a min and max value that is inside the list.
        coef_onsite (list): Alternative to coef_range.
        coef_couplings(list): Alternative to coef_range. List with the coupling values J_ij
        seed (int): Generate the same random values of J_ij.

    Returns:
       array: Array with the couplings constants J_ij
    """
    np.random.seed(seed)

    # Randomly generate the values of J_ij
    if coef_range:
        if len(coef_range) != 2:
            raise ValueError(
                "Amp range must contain only 2 values. The first value must be the lowest and the second value the highest to be generated."
            )
        if coef_onsites or coef_couplings:
            raise NameError(
                "Coefficients error. Random and custom coefficients detected, decide which one you want to use"
            )

        coefficients = np.random.uniform(
            low=coef_range[0], high=coef_range[1], size=n_particles * (n_particles + 1) // 2
        )
        coef_onsites = coefficients[:n_particles]
        coef_couplings = coefficients[n_particles:]

    # Customize the values of J_ij
    elif coef_onsites and coef_couplings:
        _ = np.random.uniform(low=0, high=1, size=n_particles * (n_particles + 1) // 2)
        length_coef_coupling = n_particles * (n_particles + 1) // 2 - n_particles
        if len(coef_onsites) != n_particles:
            raise ValueError(
                f"The length of len(coef_onsite)={len(coef_onsites)}. The expected length is {n_particles}"
            )
        if len(coef_couplings) != (length_coef_coupling):
            raise ValueError(
                f"The length of len(coef_coupling)={len(coef_couplings)}. The expected length is {length_coef_coupling}"
            )

    else:
        raise NameError("Either coef_range or coef_onsite, coef_coupling (both) must be defined")

    return (coef_onsites, coef_couplings)


def get_evolution_op(hamiltonian, dt, raising_error=True):
    """The evolution operator is defined as e^{-iHΔt}. To compute the matrix exponential
    of H, we use the function expm from scipy package.

    Args:
        hamiltonian (sparse.csc): Matrix describing how the system will evolve.
        dt (int): Period of time the system is evolving.

    Raises:
        ValueError: When the evolution_op (e^{-iHΔt}) is not unitary (UU^\dagger != I)

    Returns:
        (np.array): Unitary evolution of the system for a period of time dt
    """

    evolution_op = expm(-1.0j * hamiltonian * dt)

    if raising_error:
        if not is_unitary(evolution_op.toarray()):
            raise ValueError("Evolution operator is not unitary")

    return evolution_op.toarray()


def spin_plus(site):
    """S+ = Sx + iSy where Sx = 1/2 [[0, 1], [1, 0]] and Sy = 1/2 [[0, -i], [i, 0]]
    By definition S+ (S+ = [[0, 1], [0, 0]]) corresponds to the fermionic annihilation operator

    Args:
        site (int): From N particles the spin_plus will act on site i.

    Returns:
        QubitOperator: Math representation of the S+
    """
    Sx = spin_x(site)
    Sy = spin_y(site)
    return Sx + 1.0j * Sy


def spin_minus(site):
    """S- = Sx - iSy where Sx = 1/2 [[0, 1], [1, 0]] and Sy = 1/2 [[0, -i], [i, 0]]
    By definition S- (S- = [[0, 0], [1, 0]]) corresponds to the fermionic creation operator

    Args:
        site (int): From N particles the spin_plus will act on site i.

    Returns:
        QubitOperator: Math representation of the S-
    """
    Sx = spin_x(site)
    Sy = spin_y(site)
    return Sx - 1.0j * Sy


def spin_x(site):
    return QubitOperator((site, "X"), 1 / 2)


def spin_y(site):
    return QubitOperator((site, "Y"), 1 / 2)


def is_hermitian(matrix):
    return (matrix == matrix.H).all()
