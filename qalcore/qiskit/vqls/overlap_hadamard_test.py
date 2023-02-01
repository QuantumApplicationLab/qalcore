"""Hadammard test."""

from typing import Optional, List, Union
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


class OverlapHadammardTest:
    r"""Class to compute the Hadamard Test
    """
    def __init__(
        self,
        operators: List[QuantumCircuit],
        use_barrier: Optional[bool] = False,
        apply_initial_state: Optional[QuantumCircuit] = None,
        apply_measurement: Optional[bool] = False,
    ) -> List[QuantumCircuit]:
        r"""Create the quantum circuits required to compute the hadamard test:

        .. math::

            \\langle 0 | U^\dagger A_l V | 0 \\rangle \\langle V^\dagger A_m^\dagger U | 0 \\rangle

        Args:
            operators (List[QuantumCircuit]): List of quantum circuits representing the operators [U, A_l, A_m].
            use_barrier (Optional[bool], optional): introduce barriers in the description of the circuits.  Defaults to False.
            apply_control_to_operator (Optional[bool], optional): Apply control operator to the input quantum circuits. Defaults to True.
            apply_initial_state (Optional[QuantumCircuit], optional): Quantum Circuit to create |Psi> from |0>. If None, assume that the qubits of the firsr register are alredy in Psi.
            apply_measurement (Optional[bool], optional): apply explicit measurement. Defaults to False.

        Returns:
            List[QuantumCircuit]: List of quamtum circuits required to compute the Hadammard Test.
        """



        self.num_qubits = 2*operators[0].num_qubits + 1
        if apply_initial_state is not None:
            if apply_initial_state.num_qubits != operators[0].num_qubits:
                raise ValueError(
                    "The operator and the initial state circuits have different numbers of qubits"
                )


        # classical bit for explicit measurement
        self.num_clbits = self.num_qubits

        # build the circuits
        self.circuits = self._build_circuit(
            operators,
            use_barrier,
            apply_initial_state,
            apply_measurement,
        )

        # number of circuits required
        self.ncircuits = len(self.circuits)

        # var for iterator
        self.iiter = None

    def __iter__(self):
        self.iiter = 0
        return self

    def __next__(self):
        if self.iiter < self.ncircuits:
            out = self.circuits[self.iiter]
            self.iiter += 1
            return out
        raise StopIteration

    def __len__(self):
        return len(self.circuits)

    def __getitem__(self, index):
        return self.circuits[index]

    def _build_circuit(
        self,
        operators: List[QuantumCircuit],
        use_barrier: bool,
        apply_initial_state: Optional[QuantumCircuit] = None,
        apply_measurement: Optional[bool] = False,
    ) -> List[QuantumCircuit]:
        """build the quantum circuits

        Args:
            operators (List[QuantumCircuit]): quantum circuit or list of quantum circuits representing the [U, Al, Am].
            use_barrier (bool): introduce barriers in the description of the circuits.
            apply_initial_state (Optional[QuantumCircuit], optional): Quantum Circuit to create |Psi> from |0>. If None, assume that the qubits are alredy in Psi.  Defaults to None.
            apply_measurement (Optional[bool], optional): apply explicit measurement. Defaults to False.

        Returns:
            List[QuantumCircuit]: List of quamtum circuits required to compute the Hadammard Test.
        """

        circuits = []
        U, Al, Am = operators

        for imaginary in [False, True]:

            qctrl = QuantumRegister(1, 'qctrl')
            qreg0 = QuantumRegister(Al.num_qubits, 'qr0')
            qreg1 = QuantumRegister(Am.num_qubits, 'qr1')
            qc = QuantumCircuit(qctrl, qreg0, qreg1)

            # hadadmard gate on ctrl qbit
            qc.h(qctrl)

            # prepare psi on the first register
            if apply_initial_state is not None:
                qc.compose(
                    apply_initial_state, qreg0[:], inplace=True
                )

            # apply U on the second register
            qc.compose(U, qreg1, inplace=True)

            if use_barrier:
                qc.barrier()

            # apply Al on the first qreg
            qc.compose(Al, qreg0, inplace=True)

            # apply Am^\dagger on the second reg
            qc.compose(Am.inverse, qreg1, inplace=True)

            if use_barrier:
                qc.barrier()

            # apply the cnot gate
            for q0, q1 in zip(qreg0, qreg1):
                qc.cx(q0,q1)
            
            # Sdg on ctrl qbit
            if imaginary:
                qc.rz(qctrl)

            if use_barrier:
                qc.barrier()

            # hadamard on ctrl circuit
            qc.h(qctrl)
            for q0 in qreg0:
                qc.h(q0) 

            # measure
            if apply_measurement:
                qc.measure_all(inplace=True)
                
            circuits.append(qc)

        return circuits
