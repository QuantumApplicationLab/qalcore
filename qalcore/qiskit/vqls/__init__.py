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

# The code of the vqls has been moved to: https://github.com/QuantumApplicationLab/vqls-prototype 
# and has been made available through the qiskit ecosystem : https://qiskit.org/ecosystem/

from vqls_prototype.solver.vqls import VQLS
from vqls_prototype.solver.qst_vqls import QST_VQLS
from vqls_prototype.solver.hybrid_qst_vqls import Hybrid_QST_VQLS


__all__ = [
    "VQLS",
    "QST_VQLS",
    "Hybrid_QST_VQLS"
]
