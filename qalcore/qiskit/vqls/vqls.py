# Variational Quantum Linear Solver
# Ref : 
# Tutorial :

import numpy as np
import qiskit 
from qiskit.circuit.library.n_local.real_amplitudes import RealAmplitudes
from qiskit import Aer, transpile, assemble

from qalcore.utils.unitary_decomposition import UnitaryDecomposition
from qalcore.qiskit.utils.circ_utils import vector_creation, get_controlled_matrix

class VQLS:

    def __init__(self, A, b):
        """Variational Quantum Linear Solver 

        .. math
            A x = b

        Args:
            A (np.array): matrix of the linear system
            b (np.array): rhs of the system
        """

        self.A = A
        self.b = b 

        self.system_size = self.A.shape[0]
        self.nqbit = np.log2(self.system_size)

    def solve(self, ansatz=None, iter=100, backend=Aer.get_backend('aer_simulator')):

        # decompose the A matrix as a sum of unitary matrices
        unitdecomp_A = UnitaryDecomposition(self.A)
        unitdecomp_A.decompose(checl=True)

        # get the matrix of the cricuit needed to create the rhs
        norm, Ub_mat = vector_creation(self.b, self.nqbit, decimals=6)

        # get the controlled version of the matrices
        ctrl_Ub_mat = get_controlled_matrix([Ub_mat], 0, [1,2,3])[0].real
        ctrl_Ub_mat_dagger = get_controlled_matrix([Ub_mat.transpose()], 0, [1,2,3])[0].real

        # variational ansatz
        if ansatz is None:
            ansatz = RealAmplitudes(self.nqbit, entanglement='linear', reps=2, insert_barriers=True)
        else:
            assert(type(ansatz)==qiskit.QuantumCircuit)



