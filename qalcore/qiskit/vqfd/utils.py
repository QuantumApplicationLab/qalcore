"""Prepare the state 
|0>|A> + |1>|B>."""

from curses import meta
from typing import Optional, List, Union, Dict, Sequence
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit.register import Register
from qiskit.circuit.bit import Bit

class HadamardCircuitSuperposition(QuantumCircuit):

    def __init__(
        self,
        num_qubit: int,
        circuit_A: QuantumCircuit,
        circuit_B: QuantumCircuit,
        name: Optional[str] = None,
        global_phase: ParameterValueType = 0,
        metadata: Optional[Dict] = None,
    ):

        super().__init__(num_qubit)

        self.h(0)
        self.compose(circuit_A.control(1), qubits=list(range(0, self.num_qubits)), inplace=True)
        self.x(0)
        self.compose(circuit_B.control(1), qubits=list(range(0, self.num_qubits)), inplace=True)


class ShiftOperator(QuantumCircuit):

    def __init__(
        self,
        regs: Union[Register, int, Sequence[Bit]],
        name: Optional[str] = None,
        global_phase: ParameterValueType = 0,
        metadata: Optional[Dict] = None,
        use_mct_ancilla: bool = False
    ):

        self.qreg = QuantumRegister(regs)
        super().__init__(self.qreg)
        

        if not use_mct_ancilla:
            for i in reversed(range(1, self.num_qubits)):
                self.mct(self.qreg[:i], self.qreg[i])
            self.x(self.qreg[0])
        else:
            qreg_shift_ancilla = QuantumRegister(self.num_qubits-3, 'q_shift_ancilla')
            self.add_register(qreg_shift_ancilla)
            for i in reversed(range(1, self.num_qubits)):
                self.mct(self.qreg[:i], self.qreg[i], qreg_shift_ancilla, mode='v-chain')
            self.x(self.qreg[0])