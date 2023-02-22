# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
=======================================
Variational Quantum Linear Solver
=======================================
"""

from qalcore.qiskit.vqls.vqls import VQLS, VQLSLog
from qalcore.qiskit.vqls.hadamard_test import HadammardTest, HadammardOverlapTest
from qalcore.qiskit.vqls.numpy_unitary_matrices import UnitaryDecomposition
from qalcore.qiskit.vqls.variational_linear_solver import (
    VariationalLinearSolver,
    VariationalResult,
)

__all__ = [
    "VQLS",
    "VQLSLog"
    "HadammardTest",
    "HadammardOverlapTest",
    "UnitaryDecomposition",
    "VariationalLinearSolver",
    "VariationalResult",
]
