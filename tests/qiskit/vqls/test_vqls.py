# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test VQLS """


import unittest
from qiskit.test import QiskitTestCase

import numpy as np
# from qiskit.algorithms.linear_solvers.numpy_linear_solver import NumPyLinearSolver

from qiskit import BasicAer, QuantumCircuit
from qiskit.circuit.library import RealAmplitudes

from qiskit.quantum_info import Statevector

from qiskit.utils import QuantumInstance, algorithm_globals, has_aer
from qiskit.circuit.library.n_local.real_amplitudes import RealAmplitudes
from qalcore.qiskit.vqls.numpy_unitary_matrices import UnitaryDecomposition

from qiskit.quantum_info import Operator
from qiskit.algorithms.optimizers import COBYLA

from qalcore.qiskit.vqls import VQLS, VQLSLog

if has_aer():
    from qiskit import Aer

class TestVQLS(QiskitTestCase):
    """Test VQLS"""

    def setUp(self):
        super().setUp()
        self.seed = 50
        algorithm_globals.random_seed = self.seed
        
        self.options = (
            {
                "use_local_cost_function": False,
                "use_overlap_test": False
            },
            {
                "use_local_cost_function": True,
                "use_overlap_test": False,
            },
            {
                "use_local_cost_function": False,
                "use_overlap_test": True
            },             
        )

        self.backends = (
            QuantumInstance(
                BasicAer.get_backend("qasm_simulator"),
                shots=1024,
                seed_simulator=self.seed,
                seed_transpiler=self.seed,
            ),

            QuantumInstance(
                BasicAer.get_backend("statevector_simulator"),
                shots=1,
                seed_simulator=self.seed,
                seed_transpiler=self.seed,
            )
        )

        self.log = VQLSLog([],[])

    def test_numpy_input(self):
        """Test the VQLS on matrix input using statevector simulator."""
        
        matrix = np.array([ [0.50, 0.25, 0.10, 0.00],
                            [0.25, 0.50, 0.25, 0.10],
                            [0.10, 0.25, 0.50, 0.25],
                            [0.00, 0.10, 0.25, 0.50] ])

        rhs = np.array([0.1]*4)
        ansatz = RealAmplitudes(num_qubits=2, reps=3, entanglement='full')

        # classical_solution = NumPyLinearSolver().solve(matrix, rhs/np.linalg.norm(rhs))
        
        for qi in self.backends:
            for opt in self.options:
                vqls = VQLS(
                    ansatz=ansatz,
                    optimizer=COBYLA(maxiter=2, disp=True),
                    quantum_instance=qi,
                    callback=self.log.update,
                    use_local_cost_function=opt["use_local_cost_function"],
                    use_overlap_test=opt["use_overlap_test"]
                )
                res = vqls.solve(matrix, rhs)

                # ref_solution = np.abs(classical_solution.state / np.linalg.norm(classical_solution.state))
                # vqls_solution = np.abs(np.real(Statevector(res.state).data))
                
                # with self.subTest(msg="test solution"):
                #     assert np.allclose(ref_solution, vqls_solution, atol=1E-1, rtol=1E-1)


    def test_circuit_input_statevector(self):
        """Test the VQLS on circuits input using statevector simulator."""

        num_qubits = 2
        ansatz = RealAmplitudes(num_qubits=num_qubits, reps=3, entanglement='full')

        rhs = QuantumCircuit(num_qubits)
        rhs.h(0)
        rhs.h(1)

        qc1 = QuantumCircuit(num_qubits)
        qc1.x(0)
        qc1.x(1)
        qc1.cnot(0,1)

        qc2 = QuantumCircuit(num_qubits)
        qc2.h(0)
        qc2.x(1)
        qc2.cnot(0,1)

        matrix = UnitaryDecomposition(
            circuits = [qc1, qc2],
            coefficients = [0.5, 0.5]
        )

        np_matrix = matrix.recompose(matrix.coefficients, matrix.unitary_matrices )
        np_rhs = Operator(rhs).data @ np.array([1,0,0,0])

        # classical_solution = NumPyLinearSolver().solve(np_matrix, np_rhs/np.linalg.norm(np_rhs))
        for qi in self.backends:
            for opt in self.options:
                vqls = VQLS(
                    ansatz=ansatz,
                    optimizer=COBYLA(maxiter=2, disp=True),
                    quantum_instance=qi,
                    callback=self.log.update,
                    use_local_cost_function=opt["use_local_cost_function"],
                    use_overlap_test=opt["use_overlap_test"]
                )
                res = vqls.solve([[0.5, qc1], [0.5, qc2]], rhs)

                # ref_solution = np.abs(classical_solution.state / np.linalg.norm(classical_solution.state))
                # vqls_solution = np.abs(np.real(Statevector(res.state).data))
                
                # with self.subTest(msg="test solution"):
                #     assert np.allclose(ref_solution, vqls_solution, atol=1E-1, rtol=1E-1)


if __name__ == "__main__":
    unittest.main()
