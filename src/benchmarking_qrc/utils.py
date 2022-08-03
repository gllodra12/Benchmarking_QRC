import numpy as np


def trace_equal_one(density_op):
    """Check if the density operator has trace 1

    Args:
        density_op (np.array): Square matrix that codifies all system information.

    Returns:
        bool: True if the density_op trace is 1, false otherwise.
    """
    return np.around(np.trace(density_op), decimals=6) == 1


def is_unitary(matrix):
    """Check if the a matrix is unitary

    Args:
        matrix (np.array): Square matrix

    Returns:
        bool: True if the matrix satisfies UU^{\dagger} = I, false otherwise
    """
    length = len(matrix)
    identity_real = (np.around(np.matmul(matrix, matrix.conj().T).real, decimals=6) == np.eye(length)).all()
    zeros_imag = (np.around(np.matmul(matrix, matrix.conj().T).imag, decimals=6) == np.zeros((length, length))).all()
    return identity_real and zeros_imag
