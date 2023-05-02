import numpy as np
from numpy.testing import assert_allclose
import pytest

from qalcore.qiskit.vqls.numpy_unitary_matrices import (
    Decomposition,
    UnitaryDecomposition,
)


@pytest.fixture(params=[2, 4, 8, 16])
def symmetric(request):
    dim = request.param
    mat = np.random.rand(dim, dim)
    mat = mat + mat.T
    return mat


@pytest.mark.skip(reason="WIP")
def test_decomposition_base():
    mat = np.eye(4)[-1::-1]
    with pytest.raises(NotImplementedError, match="decompose.+Decomposition"):
        Decomposition(matrix=mat)


def test_unitary_decomposition(symmetric):
    decomp = UnitaryDecomposition(matrix=symmetric)
    assert_allclose(decomp.recompose(), symmetric)
