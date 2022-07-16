"""Hadammard test."""


from pyparsing import Optional, List
from qiskit import QuantumCircuit


class HadammardTest(QuantumCircuit):

    def __init__(
        self,
        variational_ansatz: QuantumCircuit,
        operators: List[QuantumCircuit],
        num_qubits: Optional[int] = None,
        index_auxiliary_qubit: Optional[int] = 0,
        imaginary: Optional[bool] = False
    ):
        self.variational_ansatz = variational_ansatz
        self.operators = operators
        self.num_qubits = num_qubits
        self.index_auxiliary_qubit = index_auxiliary_qubit
        self.imaginary = imaginary

        self._build_circuit()

    def _build_circuit(self):

        self.h(self.index_auxiliary_qubit)
        if self.imaginary:
            self.sdg(self.index_auxiliary_qubit)

        self.compose(self.variational_ansatz)

        for op in self.operators:
            self.compose(op.control(self.index_auxiliary_qubit))

        self.h(self.index_auxiliary_qubit)
        
        