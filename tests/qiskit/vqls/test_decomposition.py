import numpy as np
from numpy.testing import assert_allclose
import pytest

from qalcore.qiskit.vqls import (
    MatrixDecomposition,
    SymmetricDecomposition,
    PauliDecomposition
)


@pytest.fixture(params=[2, 4, 8, 16])
def symmetric(request):
    dim = request.param
    mat = np.random.rand(dim, dim)
    mat = mat + mat.T
    return mat


def test_decomposition_raises():
    mat = np.eye(4)[-1::-1]
    with pytest.raises(NotImplementedError, match="decompose.+MatrixDecomposition"):
        MatrixDecomposition(matrix=mat)


@pytest.mark.parametrize("decomposition_t", [SymmetricDecomposition, PauliDecomposition])
def test_decomposition(symmetric, decomposition_t):
    decomp = decomposition_t(matrix=symmetric)
    assert_allclose(decomp.recompose(), symmetric)
