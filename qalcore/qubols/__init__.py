"""
=======================================
Variational Quantum Linear Solver
=======================================
"""

# The code of the vqls has been moved to: https://github.com/QuantumApplicationLab/qubols


from qubols.qubols import QUBOLS
from qubols.encodings import (
    EfficientEncoding,
    RealQbitEncoding,
    RealUnitQbitEncoding,
    PositiveQbitEncoding,
)


__all__ = [
    "QUBOLS",
    "EfficientEncoding",
    "RealQbitEncoding",
    "RealUnitQbitEncoding",
    "PositiveQbitEncoding",
]
