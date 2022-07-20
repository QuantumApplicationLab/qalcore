"""Hadammard test."""


from typing import Optional, List
from qiskit import QuantumCircuit


class HadammardTest(QuantumCircuit):

    def __init__(
        self,
        ansatz: QuantumCircuit,
        operators: List[QuantumCircuit],
        num_qubits: Optional[int] = None,
        imaginary: Optional[bool] = False
    ):
        self.variational_ansatz = ansatz
        self.operators = operators
        # self.num_qubits = num_qubits
        self.imaginary = imaginary
        super().__init__(num_qubits, 
                         name='hdmr',
                         metadata={'description':'Special Hadammard Test Ref : []'})
                         
        self._build_circuit()

    def _build_circuit(self):

        # hadadmard gate on ctrl qbit
        self.h(0)

        # Sdg on ctrl qbit
        if self.imaginary:
            self.sdg(0)
        
        self.barrier()

        # ansatz
        self.compose(self.variational_ansatz, 
                     qubits=list(range(1,self.num_qubits)),
                     inplace=True)

        # matrix circuit
        for op in self.operators:
            self.compose(op.control(1),
                         qubits=list(range(0,self.num_qubits)),
                         inplace=True)

        # hadamard on ctrl circuit
        self.h(0)
        
        