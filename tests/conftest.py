import numpy as np
import pytest


@pytest.fixture
def custom_hamiltonian(op):
    """We choose the same random seed, as in hamiltonian.py to test if the code works as expected.

    Args:
        op (dict): Either bosonic_op or fermionic_op are the expected dicts

    Returns:
        sympy.matrices: Hamiltonian for a bosonic/fermionic system of 2 particles.
    """
    np.random.seed(3)
    J_11, J_12, J_22 = np.random.uniform(low=0, high=1, size=3)
    return J_11 * op["a1^dag a1"] + J_12 * op["a1^dag a2"] + J_12 * op["a2^dag a1"] + J_22 * op["a2^dag a2"]
