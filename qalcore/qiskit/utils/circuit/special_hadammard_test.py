"""Special Hadammard test."""


from pyparsing import Optional, List
from qiskit import QuantumCircuit
from qalcore.qiskit.utils.circuit.hadammard_test import HadammardTest

class SpecialHadammardTest(HadammardTest):

    def __init__(
        self,
        variational_ansatz: QuantumCircuit,
        operators: List[QuantumCircuit],
        num_qubits: Optional[int] = None,
        index_auxiliary_qubit: Optional[int] = 0,
        imaginary: Optional[bool] = False
    ):

        variational_ansatz = variational_ansatz.control(index_auxiliary_qubit)
        super().__init__(
            variational_ansatz=variational_ansatz,
            operators=operators,
            num_qubits=num_qubits,
            index_auxiliary_qubit=index_auxiliary_qubit,
            imaginary=imaginary
        )

    
