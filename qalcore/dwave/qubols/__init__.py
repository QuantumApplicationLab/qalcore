"""
=======================================
QUBO Linear Solver
=======================================
"""

# The code of the vqls has been moved to: https://github.com/QuantumApplicationLab/QUBOLS

from qubols.qubols import QUBOLS
from qubols.encodings import (
    RealQbitEncoding,
    RealUnitQbitEncoding,
    EfficientEncoding,
    PositiveQbitEncoding,
)

__all__ = [
    "QUBOLS",
    "RealUnitQbitEncoding",
    "RealQbitEncoding",
    "EfficientEncoding",
    "PositiveQbitEncoding",
]
