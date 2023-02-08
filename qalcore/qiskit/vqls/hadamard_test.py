"""Hadammard test."""

from typing import Optional, List, Union
from qiskit import QuantumCircuit, QuantumRegister
import numpy as np 

class HadammardTest:
    r"""Class to compute the Hadamard Test
    """
    def __init__(
        self,
        operators: Union[QuantumCircuit, List[QuantumCircuit]],
        use_barrier: Optional[bool] = False,
        apply_control_to_operator: Optional[Union[bool, List[bool]]] = True,
        apply_initial_state: Optional[QuantumCircuit] = None,
        apply_measurement: Optional[bool] = False,
    ) -> List[QuantumCircuit]:
        r"""Create the quantum circuits required to compute the hadamard test:

        .. math::

            \\langle \\Psi | U | \\Psi \\rangle

        Args:
            operators (Union[QuantumCircuit, List[QuantumCircuit]]): quantum circuit or list of quantum circuits representing the U.
            use_barrier (Optional[bool], optional): introduce barriers in the description of the circuits.  Defaults to False.
            apply_control_to_operator (Optional[bool], optional): Apply control operator to the input quantum circuits. Defaults to True.
            apply_initial_state (Optional[QuantumCircuit], optional): Quantum Circuit to create |Psi> from |0>. If None, assume that the qubits are alredy in Psi.
            apply_measurement (Optional[bool], optional): apply explicit measurement. Defaults to False.

        Returns:
            List[QuantumCircuit]: List of quamtum circuits required to compute the Hadammard Test.
        """

        if isinstance(operators, QuantumCircuit):
            operators = [operators]

        if not isinstance(apply_control_to_operator, list):
            apply_control_to_operator = [apply_control_to_operator]*len(operators)

        if apply_control_to_operator[0]:
            self.num_qubits = operators[0].num_qubits + 1
            if apply_initial_state is not None:
                if apply_initial_state.num_qubits != operators[0].num_qubits:
                    raise ValueError(
                        "The operator and the initial state circuits have different numbers of qubits"
                    )
        else:
            self.num_qubits = operators[0].num_qubits
            if apply_initial_state is not None:
                if apply_initial_state.num_qubits != operators[0].num_qubits - 1:
                    raise ValueError(
                        "The operator and the initial state circuits have different numbers of qubits"
                    )

        # classical bit for explicit measurement
        self.num_clbits = 1

        # build the circuits
        self.circuits = self._build_circuit(
            operators,
            use_barrier,
            apply_control_to_operator,
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
        apply_control_to_operator: bool,
        apply_initial_state: Optional[QuantumCircuit] = None,
        apply_measurement: Optional[bool] = False,
    ) -> List[QuantumCircuit]:
        """build the quantum circuits

        Args:
            operators (List[QuantumCircuit]): quantum circuit or list of quantum circuits representing the U.
            use_barrier (bool): introduce barriers in the description of the circuits.
            apply_control_to_operator (bool): Apply control operator to the input quantum circuits.
            apply_initial_state (Optional[QuantumCircuit], optional): Quantum Circuit to create |Psi> from |0>. If None, assume that the qubits are alredy in Psi.  Defaults to None.
            apply_measurement (Optional[bool], optional): apply explicit measurement. Defaults to False.

        Returns:
            List[QuantumCircuit]: List of quamtum circuits required to compute the Hadammard Test.
        """

        circuits = []

        for imaginary in [False, True]:

            if apply_measurement:
                qc = QuantumCircuit(self.num_qubits, self.num_clbits)
            else:
                qc = QuantumCircuit(self.num_qubits)

            if apply_initial_state is not None:
                qc.compose(
                    apply_initial_state, list(range(1, self.num_qubits)), inplace=True
                )

            if use_barrier:
                qc.barrier()

            # hadadmard gate on ctrl qbit
            qc.h(0)

            # Sdg on ctrl qbit
            if imaginary:
                qc.sdg(0)

            if use_barrier:
                qc.barrier()

            # matrix circuit
            for op, ctrl in zip(operators, apply_control_to_operator):
                if ctrl:
                    qc.compose(
                        op.control(1),
                        qubits=list(range(0, self.num_qubits)),
                        inplace=True,
                    )
                else:
                    qc.compose(op, qubits=list(range(0, self.num_qubits)), inplace=True)
            if use_barrier:
                qc.barrier()

            # hadamard on ctrl circuit
            qc.h(0)

            # measure
            if apply_measurement:
                qc.measure(0, 0)

            circuits.append(qc)

        return circuits


class HadammardOverlapTest:
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
            qc.compose(Am.inverse(), qreg1, inplace=True)

            if use_barrier:
                qc.barrier()

            # apply the cnot gate
            for q0, q1 in zip(qreg0, qreg1):
                qc.cx(q0,q1)
            
            # Sdg on ctrl qbit
            if imaginary:
                qc.rz(-np.pi/2, qctrl)

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

class LocalHadammardTest:
    r"""Class to compute the Hadamard Test
    """
    def __init__(
        self,
        operators: List[QuantumCircuit],
        index_zgate: int,
        apply_initial_state: Optional[QuantumCircuit] = None,
        apply_measurement: Optional[bool] = False,
    ) -> List[QuantumCircuit]:
        r"""Create the quantum circuits required to compute the hadamard test:

        .. math::

            \\langle \\Psi | U | \\Psi \\rangle

        Args:
            operators (List[QuantumCircuit]):list of quantum circuits representing the [U, Al, Am].
            index_zgate (int): index of the qubits where to z gate controlled by the ancilla qubit
            use_barrier (Optional[bool], optional): introduce barriers in the description of the circuits.  Defaults to False.
            apply_control_to_operator (Optional[bool], optional): Apply control operator to the input quantum circuits. Defaults to True.
            apply_initial_state (Optional[QuantumCircuit], optional): Quantum Circuit to create |Psi> from |0>. If None, assume that the qubits are alredy in Psi.
            apply_measurement (Optional[bool], optional): apply explicit measurement. Defaults to False.

        Returns:
            List[QuantumCircuit]: List of quamtum circuits required to compute the Hadammard Test.
        """

        if isinstance(operators, QuantumCircuit):
            operators = [operators]

        self.num_qubits = operators[0].num_qubits + 1
        if apply_initial_state is not None:
            if apply_initial_state.num_qubits != operators[0].num_qubits:
                raise ValueError(
                    "The operator and the initial state circuits have different numbers of qubits"
                )

        # classical bit for explicit measurement
        self.num_clbits = 1

        # build the circuits
        self.circuits = self._build_circuit(
            operators,
            index_zgate,
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
        index_zgate,
        apply_initial_state: Optional[QuantumCircuit] = None,
        apply_measurement: Optional[bool] = False,
    ) -> List[QuantumCircuit]:
        """build the quantum circuits

        Args:
            operators (List[QuantumCircuit]): quantum circuit or list of quantum circuits representing the U.
            use_barrier (bool): introduce barriers in the description of the circuits.
            apply_control_to_operator (bool): Apply control operator to the input quantum circuits.
            apply_initial_state (Optional[QuantumCircuit], optional): Quantum Circuit to create |Psi> from |0>. If None, assume that the qubits are alredy in Psi.  Defaults to None.
            apply_measurement (Optional[bool], optional): apply explicit measurement. Defaults to False.

        Returns:
            List[QuantumCircuit]: List of quamtum circuits required to compute the Hadammard Test.
        """

        circuits = []
        U, Al, Am = operators
        for imaginary in [False, True]:

            if apply_measurement:
                qc = QuantumCircuit(self.num_qubits, self.num_clbits)
            else:
                qc = QuantumCircuit(self.num_qubits)

            if apply_initial_state is not None:
                qc.compose(
                    apply_initial_state, list(range(1, self.num_qubits)), inplace=True
                )

            # hadadmard gate on ctrl qbit
            qc.h(0)

            # Sdg on ctrl qbit
            if imaginary:
                qc.sdg(0)

            # Al matrix circuit
            qc.compose(
                Al.control(1),
                qubits=list(range(0, self.num_qubits)),
                inplace=True,
            )

            # U* matrix
            qc.compose(
                U.inverse(),
                qubits=list(range(1, self.num_qubits)),
                inplace=True,
            )

            # control z gate on the iq qubit
            qc.cz(0, index_zgate+1)

            # U matrix
            qc.compose(
                U,
                qubits=list(range(1, self.num_qubits)),
                inplace=True,
            )

            # Am* matrix circuit
            qc.compose(
                Am.inverse().control(1),
                qubits=list(range(0, self.num_qubits)),
                inplace=True,
            )

            # hadamard on ctrl circuit
            qc.h(0)

            # measure
            if apply_measurement:
                qc.measure(0, 0)

            circuits.append(qc)

        return circuits