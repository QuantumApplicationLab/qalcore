# Variational Quantum Linear Solver
# Ref :
# Tutorial :


"""Variational Quantum Linear Solver

See https://arxiv.org/abs/1909.05820
"""



from typing import Optional, Union, List, Callable, Tuple
import numpy as np


from qiskit.circuit.library.n_local.real_amplitudes import RealAmplitudes
from qiskit import Aer
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


from qiskit.algorithms.variational_algorithm import VariationalAlgorithm


from qiskit.providers import Backend
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.utils import QuantumInstance
from qiskit.utils.backend_utils import is_aer_provider, is_statevector_backend
from qiskit.utils.validation import validate_min

from qiskit.algorithms.linear_solvers.observables.linear_system_observable import (
    LinearSystemObservable,
)

from qiskit.algorithms.minimum_eigen_solvers.vqe import (
    _validate_bounds,
    _validate_initial_point,
)

from qiskit.opflow import (
    Z,
    I,
    StateFn,
    OperatorBase,
    TensoredOp,
    ExpectationBase,
    CircuitSampler,
    ListOp,
    ExpectationFactory,
)

from qiskit.opflow.state_fns.sparse_vector_state_fn import SparseVectorStateFn

from qiskit.algorithms.optimizers import SLSQP, Minimizer, Optimizer
from qiskit.opflow.gradients import GradientBase


from qalcore.qiskit.vqls.variational_linear_solver import (
    VariationalLinearSolver,
    VariationalLinearSolverResult,
)
from qalcore.qiskit.vqls.numpy_unitary_matrices import UnitaryDecomposition
from qalcore.qiskit.vqls.hadamard_test import HadammardTest, HadammardOverlapTest


class VQLS(VariationalAlgorithm, VariationalLinearSolver):
    r"""Systems of linear equations arise naturally in many real-life applications in a wide range
    of areas, such as in the solution of Partial Differential Equations, the calibration of
    financial models, fluid simulation or numerical field calculation. The problem can be defined
    as, given a matrix :math:`A\in\mathbb{C}^{N\times N}` and a vector
    :math:`\vec{b}\in\mathbb{C}^{N}`, find :math:`\vec{x}\in\mathbb{C}^{N}` satisfying
    :math:`A\vec{x}=\vec{b}`.

    Examples:

        .. jupyter-execute:

            from qalcore.qiskit.vqls import VQLS
            from qiskit.circuit.library.n_local.real_amplitudes import RealAmplitudes
            from qiskit.algorithms.optimizers import COBYLA
            from qiskit.algorithms.linear_solvers.numpy_linear_solver import NumPyLinearSolver
            from qiskit import Aer
            import numpy as np

            # define the matrix and the rhs
            matrix = np.random.rand(4,4)
            matrix = (matrix + matrix.T)
            rhs = np.random.rand(4)

            # number of qubits needed
            num_qubits = int(log2(A.shape[0]))

            # get the classical solution
            classical_solution = NumPyLinearSolver().solve(matrix,rhs/np.linalg.norm(rhs))

            # specify the backend
            backend = Aer.get_backend('aer_simulator_statevector')

            # specify the ansatz
            ansatz = RealAmplitudes(num_qubits, entanglement='full', reps=3, insert_barriers=False)

            # declare the solver
            vqls  = VQLS(
                ansatz=ansatz,
                optimizer=COBYLA(maxiter=200, disp=True),
                quantum_instance=backend
            )

            # solve the system
            solution = vqls.solve(matrix,rhs)

    References:

        [1] Carlos Bravo-Prieto, Ryan LaRose, M. Cerezo, Yigit Subasi, Lukasz Cincio, Patrick J. Coles
        Variational Quantum Linear Solver
        `arXiv:1909.05820 <https://arxiv.org/abs/1909.05820>`
    """

    def __init__(
        self,
        ansatz: Optional[QuantumCircuit] = None,
        optimizer: Optional[Union[Optimizer, Minimizer]] = None,
        initial_point: Optional[np.ndarray] = None,
        gradient: Optional[Union[GradientBase, Callable]] = None,
        expectation: Optional[ExpectationBase] = None,
        include_custom: Optional[bool] = False,
        max_evals_grouped: Optional[int] = 1,
        callback: Optional[Callable[[int, np.ndarray, float, float], None]] = None,
        quantum_instance: Optional[Union[Backend, QuantumInstance]] = None,
        use_overlap_test: Optional[bool] = False,
        use_local_cost_function: Optional[bool] = False,
    ) -> None:
        r"""
        Args:
            ansatz: A parameterized circuit used as Ansatz for the wave function.
            optimizer: A classical optimizer. Can either be a Qiskit optimizer or a callable
                that takes an array as input and returns a Qiskit or SciPy optimization result.
            initial_point: An optional initial point (i.e. initial parameter values)
                for the optimizer. If ``None`` then VQE will look to the ansatz for a preferred
                point and if not will simply compute a random one.
            gradient: An optional gradient function or operator for optimizer.
            expectation: The Expectation converter for taking the average value of the
                Observable over the ansatz state function. When ``None`` (the default) an
                :class:`~qiskit.opflow.expectations.ExpectationFactory` is used to select
                an appropriate expectation based on the operator and backend. When using Aer
                qasm_simulator backend, with paulis, it is however much faster to leverage custom
                Aer function for the computation but, although VQE performs much faster
                with it, the outcome is ideal, with no shot noise, like using a state vector
                simulator. If you are just looking for the quickest performance when choosing Aer
                qasm_simulator and the lack of shot noise is not an issue then set `include_custom`
                parameter here to ``True`` (defaults to ``False``).
            include_custom: When `expectation` parameter here is None setting this to ``True`` will
                allow the factory to include the custom Aer pauli expectation.
            max_evals_grouped: Max number of evaluations performed simultaneously. Signals the
                given optimizer that more than one set of parameters can be supplied so that
                potentially the expectation values can be computed in parallel. Typically this is
                possible when a finite difference gradient is used by the optimizer such that
                multiple points to compute the gradient can be passed and if computed in parallel
                improve overall execution time. Deprecated if a gradient operator or function is
                given.
            callback: a callback that can access the intermediate data during the optimization.
                Three parameter values are passed to the callback as follows during each evaluation
                by the optimizer for its current set of parameters as it works towards the minimum.
                These are: the evaluation count, the cost and the optimizer parameters for the ansatz
            quantum_instance: Quantum Instance or Backend
            use_overlap_test: Use Hadamard overlap test to compute the cost function
            use_local_cost_function: use the local cost function and not the global one
        """
        super().__init__()

        validate_min("max_evals_grouped", max_evals_grouped, 1)

        self._num_qubits = None

        self._max_evals_grouped = max_evals_grouped
        self._circuit_sampler = None  # type: Optional[CircuitSampler]
        self._include_custom = include_custom

        self._ansatz = None
        self.ansatz = ansatz

        self._initial_point = None
        self.initial_point = initial_point

        self._optimizer = None
        self.optimizer = optimizer

        self._gradient = None
        self.gradient = gradient

        self._quantum_instance = None

        if quantum_instance is None:
            quantum_instance = Aer.get_backend("aer_simulator_statevector")
        self.quantum_instance = quantum_instance

        self._expectation = None
        self.expectation = expectation

        self._callback = None
        self.callback = callback

        self._eval_count = 0

        self.vector_circuit = None
        self.matrix_circuits = None
        self.num_hdr = None
        self.observable = None

        self._use_overlap_test = None 
        self.use_overlap_test = use_overlap_test

        self._use_local_cost_function = None
        self.use_local_cost_function = use_local_cost_function

        if use_local_cost_function and use_overlap_test:
            raise ValueError("Hadammard Overlap Tests not supported with local cost function")

    @property
    def num_qubits(self) -> int:
        """return the numner of qubits"""
        return self._num_qubits

    @num_qubits.setter
    def num_qubits(self, num_qubits: int) -> None:
        """Set the number of qubits"""
        self._num_qubits = num_qubits

    @property
    def num_clbits(self) -> int:
        """return the numner of classical bits"""
        return self._num_clbits

    @num_clbits.setter
    def num_clbits(self, num_clbits: int) -> None:
        """Set the number of classical bits"""
        self._num_clbits = num_clbits

    @property
    def ansatz(self) -> QuantumCircuit:
        """Returns the ansatz."""
        return self._ansatz

    @ansatz.setter
    def ansatz(self, ansatz: Optional[QuantumCircuit]):
        """Sets the ansatz.

        Args:
            ansatz: The parameterized circuit used as an ansatz.
            If None is passed, RealAmplitudes is used by default.

        """
        if ansatz is None:
            ansatz = RealAmplitudes()

        self._ansatz = ansatz
        self.num_qubits = ansatz.num_qubits + 1

    @property
    def quantum_instance(self) -> Optional[QuantumInstance]:
        """Returns quantum instance."""
        return self._quantum_instance

    @quantum_instance.setter
    def quantum_instance(
        self, quantum_instance: Union[QuantumInstance, Backend]
    ) -> None:
        """Sets quantum_instance"""
        if not isinstance(quantum_instance, QuantumInstance):
            quantum_instance = QuantumInstance(quantum_instance)

        self._quantum_instance = quantum_instance
        self._circuit_sampler = CircuitSampler(
            quantum_instance,
            statevector=is_statevector_backend(quantum_instance.backend),
            param_qobj=is_aer_provider(quantum_instance.backend),
        )

    @property
    def initial_point(self) -> Optional[np.ndarray]:
        """Returns initial point"""
        return self._initial_point

    @initial_point.setter
    def initial_point(self, initial_point: np.ndarray):
        """Sets initial point"""
        self._initial_point = initial_point

    @property
    def max_evals_grouped(self) -> int:
        """Returns max_evals_grouped"""
        return self._max_evals_grouped

    @max_evals_grouped.setter
    def max_evals_grouped(self, max_evals_grouped: int):
        """Sets max_evals_grouped"""
        self._max_evals_grouped = max_evals_grouped
        self.optimizer.set_max_evals_grouped(max_evals_grouped)

    @property
    def callback(self) -> Optional[Callable[[int, np.ndarray, float, float], None]]:
        """Returns callback"""
        return self._callback

    @callback.setter
    def callback(
        self, callback: Optional[Callable[[int, np.ndarray, float, float], None]]
    ):
        """Sets callback"""
        self._callback = callback

    @property
    def optimizer(self) -> Optimizer:
        """Returns optimizer"""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Optional[Optimizer]):
        """Sets the optimizer attribute.

        Args:
            optimizer: The optimizer to be used. If None is passed, SLSQP is used by default.

        """
        if optimizer is None:
            optimizer = SLSQP()

        if isinstance(optimizer, Optimizer):
            optimizer.set_max_evals_grouped(self.max_evals_grouped)

        self._optimizer = optimizer

    @property
    def use_overlap_test(self) -> bool:
        """return the choice for overlap hadammard test"""
        return self._use_overlap_test

    @use_overlap_test.setter
    def use_overlap_test(self, use_overlap_test: bool) -> None:
        """Set the choice for using overlap hadammard test"""
        self._use_overlap_test = use_overlap_test

    @property
    def use_local_cost_function(self) -> bool:
        """return type of cost function"""
        return self._use_local_cost_function

    @use_local_cost_function.setter
    def use_local_cost_function(self, use_local_cost_function: bool) -> None:
        """Set the type of cost function"""
        self._use_local_cost_function = use_local_cost_function

    def construct_circuit(
        self,
        matrix: Union[np.ndarray, QuantumCircuit, List],
        vector: Union[np.ndarray, QuantumCircuit],
        appply_explicit_measurement: Optional[bool] = False,
    ) -> List[QuantumCircuit]:
        """Returns the a list of circuits required to compute the expectation value

        Args:
            matrix (Union[np.ndarray, QuantumCircuit, List]): matrix of the linear system
            vector (Union[np.ndarray, QuantumCircuit]): rhs of thge linear system
            appply_explicit_measurement (bool, Optional): add the measurement operation in the circuits

        Raises:
            ValueError: if vector and matrix have different size
            ValueError: if vector and matrix have different numner of qubits
            ValueError: the input matrix is not a numoy array nor a quantum circuit

        Returns:
            List[QuantumCircuit]: Quantum Circuits required to compute the cost function
        """

        # state preparation
        if isinstance(vector, QuantumCircuit):
            nb = vector.num_qubits
            self.vector_circuit = vector

        elif isinstance(vector, np.ndarray):

            # ensure the vector is double
            vector = vector.astype("float64")

            # create the circuit
            nb = int(np.log2(len(vector)))
            self.vector_circuit = QuantumCircuit(nb)

            # prep the vector if its norm is non nul
            vec_norm = np.linalg.norm(vector)
            if vec_norm != 0:
                self.vector_circuit.prepare_state(vector / vec_norm)

        # general numpy matrix
        if isinstance(matrix, np.ndarray):

            # ensure the matrix is double
            matrix = matrix.astype("float64")

            if matrix.shape[0] != 2**self.vector_circuit.num_qubits:
                raise ValueError(
                    "Input vector dimension does not match input "
                    "matrix dimension! Vector dimension: "
                    + str(self.vector_circuit.num_qubits)
                    + ". Matrix dimension: "
                    + str(matrix.shape[0])
                )
            self.matrix_circuits = UnitaryDecomposition(matrix=matrix)

        # a single circuit
        elif isinstance(matrix, QuantumCircuit):
            if matrix.num_qubits != self.vector_circuit.num_qubits:
                raise ValueError(
                    "Matrix and vector circuits have different numbers of qubits."
                )
            self.matrix_circuits = UnitaryDecomposition(circuits=matrix)

        elif isinstance(matrix, List):
            assert isinstance(matrix[0][0], (float, complex))
            assert isinstance(matrix[0][1], QuantumCircuit)
            self.matrix_circuits = UnitaryDecomposition(
                circuits=[m[1] for m in matrix], coefficients=[m[0] for m in matrix]
            )

        else:
            raise ValueError("Format of the input matrix not recognized")

        circuits = []
        self.num_hdmr = 0

        # create only the circuit for <0|V A_n ^* A_m V|0>
        # with n != m as the diagonal terms (n==m) always give a proba of 1.0
        for ii in range(len(self.matrix_circuits)):
            mi = self.matrix_circuits[ii]

            for jj in range(ii + 1, len(self.matrix_circuits)):
                mj = self.matrix_circuits[jj]
                circuits += HadammardTest(
                    operators=[mi.circuit.inverse(), mj.circuit],
                    apply_initial_state=self._ansatz,
                    apply_measurement=appply_explicit_measurement,
                )

                self.num_hdmr += 1

        # local cost function
        if self._use_local_cost_function:

            num_z = self.matrix_circuits[0].circuit.num_qubits

            # create the circuits for <0| U^* A_l V(Zj . Ij|) V^* Am^* U|0>
            for ii in range(len(self.matrix_circuits)):
                mi = self.matrix_circuits[ii]

                for jj in range(ii, len(self.matrix_circuits)):
                    mj = self.matrix_circuits[jj]

                    for iq in range(num_z):

                        # circuit for the CZ operation on the iqth qubit
                        qc_z = QuantumCircuit(num_z+1)
                        qc_z.cz(0, iq+1)

                        # create Hadammard circuit
                        circuits += HadammardTest(
                            operators = [mi.circuit.inverse(),
                                        self.vector_circuit,
                                        qc_z,
                                        self.vector_circuit.inverse(),
                                         mj.circuit],
                            apply_control_to_operator=[True, True, False, True, True],
                            apply_initial_state = self.ansatz,
                            apply_measurement = appply_explicit_measurement 
                        )

        # global cost function 
        else:

            # create the circuits for <0|U^* A_l V|0\rangle\langle 0| V^* Am^* U|0>
            # either using overal test or hadammard test
            if self._use_overlap_test:

                for ii in range(len(self.matrix_circuits)):
                    mi = self.matrix_circuits[ii]

                    for jj in range(ii, len(self.matrix_circuits)):
                        mj = self.matrix_circuits[jj]

                        circuits += HadammardOverlapTest(
                            operators = [self.vector_circuit, mi.circuit, mj.circuit],
                            apply_initial_state = self.ansatz,
                            apply_measurement = appply_explicit_measurement 
                        )
            else:
                
                for mi in self.matrix_circuits:
                    circuits += HadammardTest(
                        operators=[self.ansatz, mi.circuit, self.vector_circuit.inverse()],
                        apply_measurement=appply_explicit_measurement,
                    )

        return circuits

    def construct_observalbe(self, num_circuits: int):
        """Create the operators needed to measure the circuit output."""

        # Create the operator to measure |1> on the control qubit.
        one_op = (I - Z) / 2
        one_op_ctrl = TensoredOp((self.num_qubits - 1) * [I]) ^ one_op

        if self._use_overlap_test:
            obs = [one_op_ctrl]*self.num_hdmr + [None] * (num_circuits-self.num_hdmr)
        else:
            obs = [one_op_ctrl]*num_circuits

        return obs

    def construct_expectation(
        self,
        parameter: Union[List[float], List[Parameter], np.ndarray],
        circuit: QuantumCircuit,
        observable: Union[TensoredOp, None],
    ) -> Union[OperatorBase, Tuple[OperatorBase, ExpectationBase]]:
        r"""
        Generate the ansatz circuit and expectation value measurement, and return their
        runnable composition.

        Args:
            parameter: Parameters for the ansatz circuit.
            circuit: one of the circuit required for the cost calculation

        Returns:
            The Operator equalling the measurement of the circuit :class:`StateFn` by the
            observable's expectation :class:`StateFn`

        """

        # assign param to circuit
        wave_function = circuit.assign_parameters(parameter)

        if observable is not None:
            # compose the statefn of the observable on the circuit
            return ~StateFn(self.observable) @ StateFn(wave_function)
        else:
            return StateFn(wave_function)



    @staticmethod
    def get_probability_from_expected_value(exp_val: complex) -> float:
        r"""Transforms the state array of the circuit into a probability

        Args:
            exp_val (complex): expected value of the observable

        Returns:
            float : probability
        """


        if isinstance(exp_val, SparseVectorStateFn):
            exp_val = exp_val.to_matrix()
            exp_val *= [1,1,1,-1]*2**()
            exp_val = exp_val.sum()

        return 1.0 - 2.0 * exp_val

    def get_hadamard_sum_coeffcients(self) -> Tuple:
        """Compute the c_i^*c_i and  c_i^*c_j coefficients.

        Returns:
            tuple: c_ii coefficients and c_ij coefficients
        """

        # compute all the ci.conj * cj  for i<j
        cii_coeffs, cij_coeffs = [], []
        for ii in range(len(self.matrix_circuits)):
            ci = self.matrix_circuits[ii].coeff
            cii_coeffs.append(ci.conj() * ci)
            for jj in range(ii + 1, len(self.matrix_circuits)):
                cj = self.matrix_circuits[jj].coeff
                cij_coeffs.append(ci.conj() * cj)

        return np.array(cii_coeffs), np.array(cij_coeffs)

    def process_probability_circuit_output(
        self, 
        probabiliy_circuit_output: List,
        coeffs: Tuple
    ) -> float:
        r"""Compute the final cost function from the output of the different circuits

        Args:
            probabiliy_circuit_output (List): expected values of the different circuits

        Returns:
            float: value of the cost function
        """

        # ci.conj * cj  for i<=j
        cii_coeffs, cij_coeffs = coeffs

        # compute all the terms in <\phi|\phi> = \sum c_i* cj <0|V Ai* Aj V|0>
        norm = self._compute_normalization_term(
            cii_coeffs, cij_coeffs, probabiliy_circuit_output
        )

        if self._use_local_cost_function:
            # compute all terms in \sum c_i* c_j 1/n \sum_n <0|V* Ai U Zn U* Aj* V|0>
            sum_terms = self._compute_local_terms(
                cii_coeffs, cij_coeffs, probabiliy_circuit_output
            )
            # add \sum c_i* cj <0|V Ai* Aj V|0>
            sum_terms += norm

            # factor two coming from |0><0| = 1/2(I+Z)
            sum_terms /= 2

        else:
            # compute all the terms in |<b|\phi>|^2 = \sum c_i* cj <0|U* Ai V|0><0|V* Aj* U|0>
            sum_terms = self._compute_global_terms(
                cii_coeffs, cij_coeffs, probabiliy_circuit_output
            )

        # overall cost
        cost = 1.0 - np.real(sum_terms / norm)

        print("Cost function %f" % cost)
        return cost

    def _compute_normalization_term(
        self,
        cii_coeff: np.ndarray,
        cij_coeff: np.ndarray,
        probabiliy_circuit_output: List,
    ) -> float:
        r"""Compute <phi|phi>

        .. math::
            \\langle\\Phi|\\Phi\\rangle = \\sum_{nm} c_n^*c_m \\langle 0|V^* U_n^* U_m V|0\\rangle

        Args:
            sum_coeff (List): the values of the c_n^* c_m coefficients
            probabiliy_circuit_output (List): the values of the circuits output

        Returns:
            float: value of the sum
        """

        # compute all the terms in <\phi|\phi> = \sum c_i* cj <0|V Ai* Aj V|0>
        norm = np.array(probabiliy_circuit_output)[: 2 * self.num_hdmr]
        if norm.dtype != "complex128":
            norm = norm.astype("complex128")
        norm *= np.array([1.0, 1.0j] * self.num_hdmr)
        norm = norm.reshape(-1, 2).sum(1)

        norm *= cij_coeff
        norm = norm.sum()
        norm += norm.conj()

        # add the diagonal terms
        # since <0|V Ai* Aj V|0> = 1 we simply
        # add the sum of the cici coeffs
        norm += cii_coeff.sum()

        return norm

    def _compute_global_terms(
        self,
        cii_coeffs: np.ndarray,
        cij_coeffs: np.ndarray,
        probabiliy_circuit_output: List,
    ) -> float:
        """Compute |<b|phi>|^2

        .. math::
            |\\langle b|\\Phi\\rangle|^2 = \\sum_{nm} c_n^*c_m \\langle 0|V^* U_n^* U_b |0 \\rangle \\langle 0|U_b^* U_m V |0\\rangle

        Args:
            cii_coeffs (List): the values of the c_i^* c_i coeffcients
            cij_coeffs (List): the values of the c_i^* c_j coeffcients
            probabiliy_circuit_output (List): values of the circuit outputs

        Returns:
            float: value of the sum
        """

        if self._use_overlap_test:

            # compute <0|V* Ai* U|0><0|U* Aj* V> = p[k] + 1.0j p[k+1]
            # with k = 2*(self.num_hdmr + f(ij))
            proba = np.array(probabiliy_circuit_output)[2 * self.num_hdmr :]
            if proba.dtype != "complex128":
                proba = proba.astype("complex128")
            proba *= np.array([1.0, 1.0j] * int(len(proba) / 2))
            proba = proba.reshape(-1, 2).sum(1)      

            # init the final result
            out = 0.0 + 0.0j
            nterm = len(proba)
            kterm, iiterm, ijterm = 0, 0, 0

            # loop over all combination of matrices
            for ii in range(len(self.matrix_circuits)):
                for jj in range(ii, len(self.matrix_circuits)):
                
                    if ii == jj:
                        # add |c_i|^2 <0|V* Ai* U|0> * <0|U* Ai V|0>
                        xii = cii_coeffs[iiterm] * proba[kterm]
                        out += xii
                        iiterm += 1

                    else:
                        # add c_i* c_j <0|V* Ai* U|0> * <0|U* Aj V|0>
                        xij = cij_coeffs[ijterm] * proba[kterm]
                        out += xij 
                        # add c_i c_j* <0|V* Aj* U|0> * <0|U* Ai V|0>
                        out += xij.conj()
                        ijterm += 1

                    kterm += 1
                    
        else:

            # compute <0|V* Ai* U|0> = p[k] + 1.0j p[k+1]
            # with k = 2*(self.num_hdmr + i)
            proba = np.array(probabiliy_circuit_output)[2 * self.num_hdmr :]
            if proba.dtype != "complex128":
                proba = proba.astype("complex128")
            proba *= np.array([1.0, 1.0j] * int(len(proba) / 2))
            proba = proba.reshape(-1, 2).sum(1)

            # init the final result
            out = 0.0 + 0.0j
            nterm = len(proba)
            iterm = 0

            for i in range(nterm):
                # add |c_i|^2 <0|V* Ai* U|0> * <0|U* Ai V|0>
                xii = cii_coeffs[i] * proba[i] * proba[i].conj()
                out += xii
                for j in range(i + 1, nterm):
                    # add c_i* c_j <0|V* Ai* U|0> * <0|U* Aj V|0>
                    xij = cij_coeffs[iterm] * proba[i] * proba[j].conj()
                    out += xij
                    # add c_j* c_i <0|V* Aj* U|0> * <0|U* Ai V|0>
                    out += xij.conj()
                    iterm += 1

        return out

    def _compute_local_terms(
        self,
        cii_coeffs: np.ndarray,
        cij_coeffs: np.ndarray,
        probabiliy_circuit_output: List,
    ) -> float:
        """Compute the term of the local cost function given by

        .. math::
            \\sum c_i^* c_j \\frac{1}{n} \\sum_n \\langle 0|V^* A_i U Z_n U^* A_j^* V|0\\rangle

        Args:
            cii_coeffs (List): the values of the c_i^* c_i coeffcients
            cij_coeffs (List): the values of the c_i^* c_j coeffcients
            probabiliy_circuit_output (List): values of the circuit outputs

        Returns:
            float: value of the sum
        """

        # rearrange the circuit output to add the real and imaginary part of the hadamard test : p[k] + 1.0j p[k+1]
        proba = np.array(probabiliy_circuit_output)[2 * self.num_hdmr :]
        if proba.dtype != "complex128":
            proba = proba.astype("complex128")
        proba *= np.array([1.0, 1.0j] * int(len(proba) / 2))
        proba = proba.reshape(-1, 2).sum(1) 

        # add all the hadamard test values corresponding to the insertion of Z gates on the same cicuit
        # b_ij = \sum_n \\frac{1}{n} \\sum_n \\langle 0|V^* A_i U Z_n U^* A_j^* V|0\\rangle
        num_zgate = self.matrix_circuits[0].circuit.num_qubits
        proba = proba.reshape(-1, num_zgate).mean(1)

        # init the final result
        out = 0.0 + 0.0j
        kterm, iiterm, ijterm = 0, 0, 0

        # loop over all combination of matrices
        for ii in range(len(self.matrix_circuits)):
            for jj in range(ii, len(self.matrix_circuits)):

            
                if ii == jj:
                    # add |c_i|^2 b_ii
                    xii = cii_coeffs[iiterm] * proba[kterm]
                    out += xii
                    iiterm += 1

                else:
                    # add c_i* c_j b_ij
                    xij = cij_coeffs[ijterm] * proba[kterm]
                    out += xij 
                    # add c_i c_j* b_ij^*
                    out += xij.conj()
                    ijterm += 1

                kterm += 1

        return out

    def get_cost_evaluation_function(
        self,
        circuits: List[QuantumCircuit],
        observables: List[Union[TensoredOp, None]],
        coeffs: Tuple,
    ) -> Callable[[np.ndarray], Union[float, List[float]]]:
        """Generate the cost function of the minimazation process

        Args:
            circuits (List[QuantumCircuit]): circuits necessary to compute the cost function

        Raises:
            RuntimeError: If the ansatz is not parametrizable

        Returns:
            Callable[[np.ndarray], Union[float, List[float]]]: the cost function
        """

        num_parameters = self.ansatz.num_parameters
        if num_parameters == 0:
            raise RuntimeError(
                "The ansatz must be parameterized, but has 0 free parameters."
            )

    
        ansatz_params = self.ansatz.parameters
        expect_ops = []
        for circ, obs in zip(circuits, observables):
            expect_ops.append(self.construct_expectation(ansatz_params, circ, obs))

        # create a ListOp for performance purposes
        expect_ops = ListOp(expect_ops)

        def cost_evaluation(parameters):

            # Create dict associating each parameter with the lists of parameterization values for it
            parameter_sets = np.reshape(parameters, (-1, num_parameters))
            param_bindings = dict(
                zip(ansatz_params, parameter_sets.transpose().tolist())
            )

            # TODO define a multiple sampler, one for each ops, to leverage caching
            # get the sampled output
            out = []
            for op in expect_ops:
                sampled_expect_op = self._circuit_sampler.convert(
                    op, params=param_bindings
                )
                out.append(
                    self.get_probability_from_expected_value(
                        sampled_expect_op.eval()[0]
                    )
                )

            # compute the total cost
            cost = self.process_probability_circuit_output(out, coeffs)

            # get the internediate results if required
            if self._callback is not None:
                for param_set in parameter_sets:
                    self._eval_count += 1
                    self._callback(self._eval_count, cost, param_set)
            else:
                self._eval_count += 1

            return cost

        return cost_evaluation

    def _calculate_observable(
        self,
        solution: QuantumCircuit,
        observable: Optional[Union[LinearSystemObservable, BaseOperator]] = None,
        observable_circuit: Optional[QuantumCircuit] = None,
        post_processing: Optional[
            Callable[[Union[float, List[float]]], Union[float, List[float]]]
        ] = None,
    ) -> Tuple[Union[float, List[float]], Union[float, List[float]]]:
        """Calculates the value of the observable(s) given.

        Args:
            solution: The quantum circuit preparing the solution x to the system.
            observable: Information to be extracted from the solution.
            observable_circuit: Circuit to be applied to the solution to extract information.
            post_processing: Function to compute the value of the observable.

        Returns:
            The value of the observable(s) and the circuit results before post-processing as a
             tuple.
        """
        # exit if nothing is provided
        if observable is None and observable_circuit is None:
            return None, None

        # Get the number of qubits
        nb = solution.num_qubits

        # if the observable is given construct post_processing and observable_circuit
        if observable is not None:
            observable_circuit = observable.observable_circuit(nb)
            post_processing = observable.post_processing

            if isinstance(observable, LinearSystemObservable):
                observable = observable.observable(nb)

        is_list = True
        if not isinstance(observable_circuit, list):
            is_list = False
            observable_circuit = [observable_circuit]
            observable = [observable]

        expectations = []
        for circ, obs in zip(observable_circuit, observable):
            circuit = QuantumCircuit(solution.num_qubits)
            circuit.append(solution, circuit.qubits)
            circuit.append(circ, range(nb))
            expectations.append(~StateFn(obs) @ StateFn(circuit))

        if is_list:
            # execute all in a list op to send circuits in batches
            expectations = ListOp(expectations)
        else:
            expectations = expectations[0]

        # check if an expectation converter is given
        if self._expectation is not None:
            expectations = self._expectation.convert(expectations)
        # if otherwise a backend was specified, try to set the best expectation value
        elif self._circuit_sampler is not None:
            if is_list:
                op = expectations.oplist[0]
            else:
                op = expectations
            self._expectation = ExpectationFactory.build(
                op, self._circuit_sampler.quantum_instance
            )

        if self._circuit_sampler is not None:
            expectations = self._circuit_sampler.convert(expectations)

        # evaluate
        expectation_results = expectations.eval()

        # apply post_processing
        result = post_processing(expectation_results, nb)

        return result, expectation_results

    def solve(
        self,
        matrix: Union[np.ndarray, QuantumCircuit, List[QuantumCircuit]],
        vector: Union[np.ndarray, QuantumCircuit],
        observable: Optional[
            Union[
                LinearSystemObservable,
                BaseOperator,
                List[LinearSystemObservable],
                List[BaseOperator],
            ]
        ] = None,
        observable_circuit: Optional[
            Union[QuantumCircuit, List[QuantumCircuit]]
        ] = None,
        post_processing: Optional[
            Callable[[Union[float, List[float]]], Union[float, List[float]]]
        ] = None,
    ) -> VariationalLinearSolverResult:
        """Solve the linear system

        Args:
            matrix (Union[List, np.ndarray, QuantumCircuit]): matrix of the linear system
            vector (Union[np.ndarray, QuantumCircuit]): rhs of the linear system
            observable: Optional information to be extracted from the solution.
                Default is `None`.
            observable_circuit: Optional circuit to be applied to the solution to extract
                information. Default is `None`.
            post_processing: Optional function to compute the value of the observable.
                Default is the raw value of measuring the observable.

        Raises:
            ValueError: If an invalid combination of observable, observable_circuit and
                post_processing is passed.

        Returns:
            VariationalLinearSolverResult: Result of the optimization and solution vector of the linear system
        """

        # compute the circuits
        circuits = self.construct_circuit(matrix, vector)

        # compute the observables needed for measuring the circiuts
        observables = self.construct_observalbe(len(circuits))

        # compute all the ci.conj * cj  for i<j
        coeffs = self.get_hadamard_sum_coeffcients()

        # set an expectation for this algorithm run (will be reset to None at the end)
        initial_point = _validate_initial_point(self.initial_point, self.ansatz)
        bounds = _validate_bounds(self.ansatz)

        # Convert the gradient operator into a callable function that is compatible with the
        # optimization routine.
        gradient = self._gradient

        self._eval_count = 0

        # get the cost evaluation function
        cost_evaluation = self.get_cost_evaluation_function(circuits, observables, coeffs)

        if callable(self.optimizer):
            opt_result = self.optimizer(  # pylint: disable=not-callable
                fun=cost_evaluation, x0=initial_point, jac=gradient, bounds=bounds
            )
        else:
            opt_result = self.optimizer.minimize(
                fun=cost_evaluation, x0=initial_point, jac=gradient, bounds=bounds
            )

        # create the solution
        solution = VariationalLinearSolverResult()

        # optimization data
        solution.optimal_point = opt_result.x
        solution.optimal_parameters = dict(zip(self.ansatz.parameters, opt_result.x))
        solution.optimal_value = opt_result.fun
        solution.cost_function_evals = opt_result.nfev

        # final ansatz
        solution.state = self.ansatz.assign_parameters(solution.optimal_parameters)

        # observable
        solution.observable = self._calculate_observable(
            solution.state, observable, observable_circuit, post_processing
        )

        return solution
