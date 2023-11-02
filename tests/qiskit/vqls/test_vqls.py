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


from qiskit.quantum_info import Operator
from qiskit.algorithms.optimizers import COBYLA
from qiskit.primitives import Estimator, Sampler, BackendEstimator, BackendSampler
from qalcore.qiskit.vqls import VQLS

if has_aer():
    from qiskit import Aer

class TestVQLS(QiskitTestCase):
    """Test VQLS"""

    def setUp(self):
        super().setUp()
        self.seed = 50
        algorithm_globals.random_seed = self.seed

        self.options = (
            {"use_local_cost_function": False, "use_overlap_test": False},
            {"use_local_cost_function": True, "use_overlap_test": False},
            {"use_local_cost_function": False, "use_overlap_test": True},
        )

        self.estimators = (
            Estimator(),
            BackendEstimator(BasicAer.get_backend("qasm_simulator")),
        )

        self.samplers = (
            Sampler(),
            BackendSampler(BasicAer.get_backend("qasm_simulator")),
        )

    def test_numpy_input(self):
        """Test the VQLS on matrix input using statevector simulator."""

        matrix = np.array(
            [
                [0.50, 0.25, 0.10, 0.00],
                [0.25, 0.50, 0.25, 0.10],
                [0.10, 0.25, 0.50, 0.25],
                [0.00, 0.10, 0.25, 0.50],
            ]
        )

        rhs = np.array([0.1] * 4)
        ansatz = RealAmplitudes(num_qubits=2, reps=3, entanglement="full")

        for estimator, sampler in zip(self.estimators, self.samplers):
            for opt in self.options:
                vqls = VQLS(
                    estimator,
                    ansatz,
                    COBYLA(maxiter=2, disp=True),
                    options=opt,
                    sampler=sampler,
                )
                _ = vqls.solve(matrix, rhs)

if __name__ == "__main__":
    unittest.main()
