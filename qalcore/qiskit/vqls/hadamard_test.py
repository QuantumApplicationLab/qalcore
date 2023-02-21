"""Hadammard test."""

from typing import Optional, List, Union
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.opflow import Z, I, TensoredOp, StateFn, ListOp
from qiskit.circuit import Parameter
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
    ) :
        r"""Create the quantum circuits required to compute the hadamard test:

        .. math::

            \\langle \\Psi | U | \\Psi \\rangle

        Args:
            operators (Union[QuantumCircuit, List[QuantumCircuit]]): quantum circuit or list of quantum circuits representing the U.
            use_barrier (Optional[bool], optional): introduce barriers in the description of the circuits.  Defaults to False.
            apply_control_to_operator (Optional[bool], optional): Apply control operator to the input quantum circuits. Defaults to True.
            apply_initial_state (Optional[QuantumCircuit], optional): Quantum Circuit to create |Psi> from |0>. If None, assume that the qubits are alredy in Psi.
            apply_measurement (Optional[bool], optional): apply explicit measurement. Defaults to False.

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

        # compute the observables
        self.observable = self._build_observable()

        # init the expectation
        self.expect_ops = None

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

    def _build_observable(self) -> List[TensoredOp]:
        """Create the operator to measure |1> on the control qubit.

        Returns:
            Lis[TensoredOp]: List of two observables to measure |1> on the control qubit I^...^I^|1><1|
        """

        one_op = (I - Z) / 2
        one_op_ctrl = TensoredOp((self.num_qubits - 1) * [I]) ^ one_op
        return one_op_ctrl

    def construct_expectation(self, parameter: Union[List[float], List[Parameter], np.ndarray, None] = None):
        r"""
        Generate the ansatz circuit and expectation value measurement, and return their
        runnable composition.

        Args:
            parameter: Parameters for the ansatz circuit.
        
        Returns:
            The Operator equalling the measurement of the circuit :class:`StateFn` by the
            observable's expectation :class:`StateFn`
        """

        exp_val = []

        for circ in self.circuits:
            if parameter is not None:
                exp_val.append(~StateFn(self.observable) @ StateFn(circ.assign_parameters(parameter)))
            else:
                exp_val.append(~StateFn(self.observable) @ StateFn(circ))

        self.expect_ops = ListOp(exp_val)

    def get_value(self, circuit_sampler, param_binding: dict) -> List:

        def post_processing(exp_val) -> float:
            return 1.0 - 2.0 * exp_val[0]

        out = []
        for op in self.expect_ops:
            sampled_val = circuit_sampler.convert(op, params=param_binding).eval()
            out.append(post_processing(sampled_val))

        out = np.array(out).astype('complex128')
        out *= np.array([1.0, 1.0j])

        return out.sum()



class HadammardOverlapTest:
    r"""Class to compute the Hadamard Test
    """
    def __init__(
        self,
        operators: List[QuantumCircuit],
        use_barrier: Optional[bool] = False,
        apply_initial_state: Optional[QuantumCircuit] = None,
        apply_measurement: Optional[bool] = False,
    ) :
        r"""Create the quantum circuits required to compute the hadamard test:

        .. math::

            \\langle 0 | U^\dagger A_l V | 0 \\rangle \\langle V^\dagger A_m^\dagger U | 0 \\rangle

        Args:
            operators (List[QuantumCircuit]): List of quantum circuits representing the operators [U, A_l, A_m].
            use_barrier (Optional[bool], optional): introduce barriers in the description of the circuits.  Defaults to False.
            apply_initial_state (Optional[QuantumCircuit], optional): Quantum Circuit to create |Psi> from |0>. If None, assume that the qubits of the firsr register are alredy in Psi.
            apply_measurement (Optional[bool], optional): apply explicit measurement. Defaults to False.

        Returns:
            List[QuantumCircuit]: List of quamtum circuits required to compute the Hadammard Test.
        """

        self.operator_num_qubits = operators[0].num_qubits 
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

        # post processing coefficients
        self.post_process_coeffs = self.compute_post_processing_coefficients()

        # var for iterator
        self.iiter = None

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
                    apply_initial_state, qreg0, inplace=True
                )

            # apply U on the second register
            qc.compose(U, qreg1, inplace=True)

            if use_barrier:
                qc.barrier()

            # apply Al on the first qreg
            idx = [0] + list(range(1,Al.num_qubits+1))
            qc.compose(Al.control(1), idx, inplace=True)

            # apply Am^\dagger on the second reg
            idx = [0] + list(range(Al.num_qubits+1,2*Al.num_qubits+1))
            qc.compose(Am.inverse().control(1), idx, inplace=True)

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

    def compute_post_processing_coefficients(self):
        """Compute the coefficients for the postprocessing 
        """

        # compute [1,1,1,-1] \otimes n
        c0 = np.array([1,1,1,-1])
        coeffs = np.array([1,1,1,-1])
        for _ in range(1,self.operator_num_qubits):
            coeffs = np.tensordot(coeffs, c0, axes=0).flatten()
        # coeffs = np.array([1,1,1,-1]*(2*2**(self.operator_num_qubits-1)))

        # create the reordering index to map
        # A0 A1 .. AN B0 B1 ... BN => A0 B0 A1 B1 ... AN BN
        bit_strings = []
        for i in range(2**(self.operator_num_qubits)):
            bit_strings.append( f"{i:b}".zfill(self.operator_num_qubits) )


        reordered_coeffs = np.zeros_like(coeffs)

        for bs1 in bit_strings:
            for bs2 in bit_strings:
                idx = int(bs1+bs2, 2)
                new_bit_string = ''.join([i+j for i,j in zip(bs1, bs2)])
                idx_ori = int(new_bit_string,2)
                reordered_coeffs[idx] = coeffs[idx_ori]

        return reordered_coeffs 

    def construct_expectation(self, parameter: Union[List[float], List[Parameter], np.ndarray, None] = None):
        r"""
        Generate the ansatz circuit and expectation value measurement, and return their
        runnable composition.

        Args:
            parameter: Parameters for the ansatz circuit.
        
        Returns:
            The Operator equalling the measurement of the circuit :class:`StateFn` by the
            observable's expectation :class:`StateFn`
        """

        exp_val = []

        for circ in self.circuits:
            if parameter is not None:
                exp_val.append(StateFn(circ.assign_parameters(parameter)))
            else:
                exp_val.append( StateFn(circ))

        self.expect_ops = ListOp(exp_val)

    def get_value(self, circuit_sampler, param_binding: dict) -> List:

        def post_processing(exp_val) -> float:
            exp_val = (exp_val.to_matrix()[0])
            exp_val = (exp_val * exp_val.conj())
            
            p0 = (exp_val[0::2] * self.post_process_coeffs).sum()
            p1 = (exp_val[1::2] * self.post_process_coeffs).sum()

            return p0 - p1

        out = []
        for op in self.expect_ops:
            sampled_val = circuit_sampler.convert(op, params=param_binding).eval()
            out.append(post_processing(sampled_val))

        out = np.array(out).astype('complex128')
        out *= np.array([1.0, 1.0j])

        return out.sum()