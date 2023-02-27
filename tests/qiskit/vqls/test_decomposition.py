import numpy as np
import pytest

from qalcore.qiskit.vqls.numpy_unitary_matrices import (
    Decomposition,
    UnitaryDecomposition,
)


def test_decomposition_base():
    mat = np.eye(4)[-1::-1]
    with pytest.raises(NotImplementedError, match="decompose.+Decomposition"):
        Decomposition(matrix=mat)


def test_unitary_decomposition():
    mat = np.eye(4)[-1::-1]
    # assert np.allclose(np.eye(16), np.dot(mat, mat.conj().T))
    decomp = UnitaryDecomposition(matrix=mat)
