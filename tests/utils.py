import numpy as np
from sympy.physics.quantum import TensorProduct
from sympy.matrices import Matrix, eye


def id(dimensions=2):
    return eye(dimensions)


def sigma_z():
    return Matrix([[1, 0], [0, -1]])


def annihil_f():
    "Annihilation operator for fermions"
    return Matrix([[0, 1], [0, 0]])


def creation_f():
    "Creation operator for fermions"
    return Matrix([[0, 0], [1, 0]])


def annihil_b():
    "Annihilation operator for a 3 level occupation boson"
    return Matrix([[0, 1, 0], [0, 0, np.sqrt(2)], [0, 0, 0]])


def creation_b():
    "Annihilation operator for a 3 level occupation boson"
    return Matrix([[0, 0, 0], [1, 0, 0], [0, np.sqrt(2), 0]])


def spin_plus():
    return Matrix([[0, 1], [0, 0]])


def spin_minus():
    return Matrix([[0, 0], [1, 0]])


fermionic_op = {
    "a1^dag a1": TensorProduct(creation_f(), id()) * TensorProduct(annihil_f(), id()),
    "a1^dag a2": TensorProduct(creation_f(), id()) * TensorProduct(sigma_z(), annihil_f()),
    "a2^dag a1": TensorProduct(sigma_z(), creation_f()) * TensorProduct(annihil_f(), id()),
    "a2^dag a2": TensorProduct(sigma_z(), creation_f()) * TensorProduct(sigma_z(), annihil_f()),
    "a1^dag a1^dag": TensorProduct(creation_f(), id()) * TensorProduct(creation_f(), id()),
    "a2^dag a2^dag": TensorProduct(sigma_z(), creation_f()) * TensorProduct(sigma_z(), creation_f()),
    "a1 a1": TensorProduct(annihil_f(), id()) * TensorProduct(annihil_f(), id()),
    "a2 a2": TensorProduct(sigma_z(), annihil_f()) * TensorProduct(sigma_z(), annihil_f()),
}

bosonic_op = {
    "a1^dag a1": TensorProduct(creation_b(), id(3)) * TensorProduct(annihil_b(), id(3)),
    "a1^dag a2": TensorProduct(creation_b(), id(3)) * TensorProduct(id(3), annihil_b()),
    "a2^dag a1": TensorProduct(id(3), creation_b()) * TensorProduct(annihil_b(), id(3)),
    "a2^dag a2": TensorProduct(id(3), creation_b()) * TensorProduct(id(3), annihil_b()),
    "a1^dag a1^dag": TensorProduct(creation_b(), id(3)) * TensorProduct(creation_b(), id(3)),
    "a2^dag a2^dag": TensorProduct(id(3), creation_b()) * TensorProduct(id(3), creation_b()),
    "a1 a1": TensorProduct(annihil_b(), id(3)) * TensorProduct(annihil_b(), id(3)),
    "a2 a2": TensorProduct(id(3), annihil_b()) * TensorProduct(id(3), annihil_b()),
}


spin_op = {
    "s1+ s1-": TensorProduct(spin_plus(), id()) * TensorProduct(spin_minus(), id()),
    "s2+ s1-": TensorProduct(id(), spin_plus()) * TensorProduct(spin_minus(), id()),
    "s1+ s2-": TensorProduct(spin_plus(), id()) * TensorProduct(id(), spin_minus()),
    "s2+ s2-": TensorProduct(id(), spin_plus()) * TensorProduct(id(), spin_minus()),
}

fermionic_op_3 = {
    "a1^dag a1": TensorProduct(creation_f(), id(4)) * TensorProduct(annihil_f(), id(4)),
    "a1^dag a2": TensorProduct(creation_f(), id(4)) * TensorProduct(sigma_z(), annihil_f(), id()),
    "a1^dag a3": TensorProduct(creation_f(), id(4)) * TensorProduct(sigma_z(), sigma_z(), annihil_f()),
    "a2^dag a1": TensorProduct(sigma_z(), creation_f(), id()) * TensorProduct(annihil_f(), id(4)),
    "a2^dag a2": TensorProduct(sigma_z(), creation_f(), id()) * TensorProduct(sigma_z(), annihil_f(), id()),
    "a2^dag a3": TensorProduct(sigma_z(), creation_f(), id()) * TensorProduct(sigma_z(), sigma_z(), annihil_f()),
    "a3^dag a1": TensorProduct(sigma_z(), sigma_z(), creation_f()) * TensorProduct(annihil_f(), id(4)),
    "a3^dag a2": TensorProduct(sigma_z(), sigma_z(), creation_f()) * TensorProduct(sigma_z(), annihil_f(), id()),
    "a3^dag a3": TensorProduct(sigma_z(), sigma_z(), creation_f()) * TensorProduct(sigma_z(), sigma_z(), annihil_f()),
    "a1^dag a1^dag": TensorProduct(creation_f(), id(4)) * TensorProduct(creation_f(), id(4)),
    "a2^dag a2^dag": TensorProduct(sigma_z(), creation_f(), id()) * TensorProduct(sigma_z(), creation_f(), id()),
    "a3^dag a3^dag": TensorProduct(sigma_z(), sigma_z(), creation_f())
    * TensorProduct(sigma_z(), sigma_z(), creation_f()),
    "a1 a1": TensorProduct(annihil_f(), id(4)) * TensorProduct(annihil_f(), id(4)),
    "a2 a2": TensorProduct(sigma_z(), annihil_f(), id()) * TensorProduct(sigma_z(), annihil_f(), id()),
    "a3 a3": TensorProduct(sigma_z(), sigma_z(), annihil_f()) * TensorProduct(sigma_z(), sigma_z(), annihil_f()),
}


bosonic_op_3 = {
    "a1^dag a1": TensorProduct(creation_b(), id(9)) * TensorProduct(annihil_b(), id(9)),
    "a1^dag a2": TensorProduct(creation_b(), id(9)) * TensorProduct(id(3), annihil_b(), id(3)),
    "a1^dag a3": TensorProduct(creation_b(), id(9)) * TensorProduct(id(9), annihil_b()),
    "a2^dag a1": TensorProduct(id(3), creation_b(), id(3)) * TensorProduct(annihil_b(), id(9)),
    "a2^dag a2": TensorProduct(id(3), creation_b(), id(3)) * TensorProduct(id(3), annihil_b(), id(3)),
    "a2^dag a3": TensorProduct(id(3), creation_b(), id(3)) * TensorProduct(id(9), annihil_b()),
    "a3^dag a1": TensorProduct(id(9), creation_b()) * TensorProduct(annihil_b(), id(9)),
    "a3^dag a2": TensorProduct(id(9), creation_b()) * TensorProduct(id(3), annihil_b(), id(3)),
    "a3^dag a3": TensorProduct(id(9), creation_b()) * TensorProduct(id(9), annihil_b()),
    "a1^dag a1^dag": TensorProduct(creation_b(), id(9)) * TensorProduct(creation_b(), id(9)),
    "a2^dag a2^dag": TensorProduct(id(3), creation_b(), id(3)) * TensorProduct(id(3), creation_b(), id(3)),
    "a3^dag a3^dag": TensorProduct(id(3), id(3), creation_b()) * TensorProduct(id(3), id(3), creation_b()),
    "a1 a1": TensorProduct(annihil_b(), id(9)) * TensorProduct(annihil_b(), id(9)),
    "a2 a2": TensorProduct(id(3), annihil_b(), id(3)) * TensorProduct(id(3), annihil_b(), id(3)),
    "a3 a3": TensorProduct(id(3), id(3), annihil_b()) * TensorProduct(id(3), id(3), annihil_b()),
}


spin_op_3 = {
    "s1+ s1-": TensorProduct(spin_plus(), id(), id()) * TensorProduct(spin_minus(), id(), id()),
    "s1+ s2-": TensorProduct(spin_plus(), id(), id()) * TensorProduct(id(), spin_minus(), id()),
    "s1+ s3-": TensorProduct(spin_plus(), id(), id()) * TensorProduct(id(), id(), spin_minus()),
    "s2+ s1-": TensorProduct(id(), spin_plus(), id()) * TensorProduct(spin_minus(), id(), id()),
    "s2+ s2-": TensorProduct(id(), spin_plus(), id()) * TensorProduct(id(), spin_minus(), id()),
    "s2+ s3-": TensorProduct(id(), spin_plus(), id()) * TensorProduct(id(), id(), spin_minus()),
    "s3+ s1-": TensorProduct(id(), id(), spin_plus()) * TensorProduct(spin_minus(), id(), id()),
    "s3+ s2-": TensorProduct(id(), id(), spin_plus()) * TensorProduct(id(), spin_minus(), id()),
    "s3+ s3-": TensorProduct(id(), id(), spin_plus()) * TensorProduct(id(), id(), spin_minus()),
}
