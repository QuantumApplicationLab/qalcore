# Variational Quantum Linear Solver
# Ref :
# Tutorial :


"""Variational Quantum Linear Solver

See https://arxiv.org/abs/1909.05820
"""



from typing import Optional, Union, List, Callable, Tuple, Dict
import numpy as np

from qiskit.circuit.library.n_local.real_amplitudes import RealAmplitudes
from qiskit import Aer
from qiskit import QuantumCircuit

from qiskit.algorithms.variational_algorithm import VariationalAlgorithm


from qiskit.providers import Backend
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.utils import QuantumInstance
from qiskit.utils.backend_utils import is_aer_provider, is_statevector_backend
from qiskit.utils.validation import validate_min


from qiskit.algorithms.minimum_eigen_solvers.vqe import (
    _validate_bounds,
    _validate_initial_point,
)

from qiskit.opflow import (
    StateFn,
    ExpectationBase,
    CircuitSampler,
    ListOp,
    ExpectationFactory,
)


from qiskit.algorithms.optimizers import SLSQP, Minimizer, Optimizer
from qiskit.opflow.gradients import GradientBase


from qalcore.qiskit.vqls.variational_linear_solver import (
    VariationalLinearSolver,
    VariationalLinearSolverResult,
)
from qalcore.qiskit.vqls.numpy_unitary_matrices import UnitaryDecomposition
from qalcore.qiskit.vqls.hadamard_test import HadammardTest, HadammardOverlapTest

from dataclasses import dataclass

@dataclass
class VQLSLog:
    values: List
    parameters: List 
    def update(self, count, cost, parameters):
        self.values.append(cost)
        self.parameters.append(parameters)
        print(f"VQLS Iteration {count} Cost {cost}", end="\r", flush=True)

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

        self.default_solve_options = {"use_overlap_test": False,
                                      "use_local_cost_function": False}

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

    def construct_circuit(
        self,
        matrix: Union[np.ndarray, QuantumCircuit, List],
        vector: Union[np.ndarray, QuantumCircuit],
        options: Dict,
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

        # create only the circuit for <psi|psi> =  <0|V A_n ^* A_m V|0>
        # with n != m as the diagonal terms (n==m) always give a proba of 1.0
        hdmr_tests_norm = self._get_norm_circuits()
        
        # create the circuits for <b|psi> 
        # local cost function
        if options["use_local_cost_function"]:
            hdmr_tests_overlap = self._get_local_circuits()

        # global cost function 
        else:
            hdmr_tests_overlap = self._get_global_circuits(options)

        return hdmr_tests_norm, hdmr_tests_overlap

    def _get_norm_circuits(self) -> List[QuantumCircuit]:
        """construct the circuit for the norm

        Raises:
            RuntimeError: _description_

        Returns:
            List[QuantumCircuit]: quantum circuits needed for the norm
        """

        hdmr_tests_norm = []

        for ii in range(len(self.matrix_circuits)):
            mi = self.matrix_circuits[ii]

            for jj in range(ii + 1, len(self.matrix_circuits)):
                mj = self.matrix_circuits[jj]
                hdmr_tests_norm.append( HadammardTest(
                    operators=[mi.circuit.inverse(), mj.circuit],
                    apply_initial_state=self._ansatz,
                    apply_measurement=False,
                    )
                )
        return hdmr_tests_norm

    def _get_local_circuits(self) -> List[QuantumCircuit]:
        """construct the circuits needed for the local cost function

        Returns:
            List[QuantumCircuit]: quantum circuit for the local cost function
        """

        hdmr_tests_overlap = []
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
                    hdmr_tests_overlap.append(HadammardTest(
                        operators = [mi.circuit,
                                    self.vector_circuit.inverse(),
                                    qc_z,
                                    self.vector_circuit,
                                    mj.circuit.inverse()],
                        apply_control_to_operator=[True, True, False, True, True],
                        apply_initial_state = self.ansatz,
                        apply_measurement = False 
                        )
                    )
        return hdmr_tests_overlap

    def _get_global_circuits(self, 
                             options: dict) -> List[QuantumCircuit]:
        """construct circuits needed for the global cost function

        Args:
            appply_explicit_measurement (bool): _description_

        Raises:
            RuntimeError: _description_

        Returns:
            List[QuantumCircuit]: _description_
        """

        hdmr_tests_overlap = []
        # create the circuits for <0|U^* A_l V|0\rangle\langle 0| V^* Am^* U|0>
        # either using overal test or hadammard test
        if options["use_overlap_test"]:

            for ii in range(len(self.matrix_circuits)):
                mi = self.matrix_circuits[ii]

                for jj in range(ii, len(self.matrix_circuits)):
                    mj = self.matrix_circuits[jj]

                    hdmr_tests_overlap.append(HadammardOverlapTest(
                        operators = [self.vector_circuit, mi.circuit, mj.circuit],
                        apply_initial_state = self.ansatz,
                        apply_measurement = False 
                        )
                    )
        else:
            
            for mi in self.matrix_circuits:
                hdmr_tests_overlap.append(HadammardTest(
                    operators=[self.ansatz, mi.circuit, self.vector_circuit.inverse()],
                    apply_measurement=False,
                    )
                )

        return hdmr_tests_overlap

    @staticmethod
    def get_coefficient_matrix(coeffs) -> np.ndarray:
        """Compute all the vi* vj terms

        Args:
            coeffs (np.ndarray): list of complex coefficients
        """
        return coeffs[:,None].conj() @ coeffs[None,:]

    def _assemble_cost_function(
        self, 
        hdmr_values_norm: np.ndarray,
        hdmr_values_overlap: np.ndarray,
        coefficient_matrix: np.ndarray,
        options: Dict
    ) -> float:
        r"""Compute the final cost function from the output of the different circuits

        Args:
            probabiliy_circuit_output (List): expected values of the different circuits

        Returns:
            float: value of the cost function
        """

        # compute all the terms in <\phi|\phi> = \sum c_i* cj <0|V Ai* Aj V|0>
        norm = self._compute_normalization_term(
            coefficient_matrix, hdmr_values_norm
        )

        if options["use_local_cost_function"]:
            # compute all terms in 
            # \sum c_i* c_j 1/n \sum_n <0|V* Ai U Zn U* Aj* V|0>
            sum_terms = self._compute_local_terms(
                coefficient_matrix, hdmr_values_overlap, norm
            )

        else:
            # compute all the terms in 
            # |<b|\phi>|^2 = \sum c_i* cj <0|U* Ai V|0><0|V* Aj* U|0>
            sum_terms = self._compute_global_terms(
                coefficient_matrix, hdmr_values_overlap, options
            )

        # overall cost
        cost = 1.0 - np.real(sum_terms / norm)

        # print("Cost function %f" % cost)
        return cost

    def _compute_normalization_term(
        self,
        coeff_matrix: np.ndarray,
        hdmr_values: np.ndarray,
    ) -> float:
        r"""Compute <phi|phi>

        .. math::
            \\langle\\Phi|\\Phi\\rangle = \\sum_{nm} c_n^*c_m \\langle 0|V^* U_n^* U_m V|0\\rangle

        Args:
            coeff_matrix (List): the matrix values of the c_n^* c_m coefficients
            hdmr_values (List): the values of the circuits output

        Returns:
            float: value of the sum
        """

        # compute all the terms in <\phi|\phi> = \sum c_i* cj <0|V Ai* Aj V|0>
        # hdrm_values here contains the values of the <0|V Ai* Aj V|0>  with j>i
        out = hdmr_values

        # we multiuply hdmrval by the triup coeff matrix and sum
        out *= coeff_matrix[np.triu_indices_from(coeff_matrix, k=1)]
        out = out.sum()

        # add the conj that corresponds to the tri down matrix
        out += out.conj()

        # add the diagonal terms
        # since <0|V Ai* Aj V|0> = 1 we simply
        # add the sum of the cici coeffs
        out += np.trace(coeff_matrix)

        return out

    def _compute_global_terms(
        self,
        coeff_matrix: np.ndarray,
        hdmr_values: np.ndarray,
        options: Dict
    ) -> float:
        """Compute |<b|phi>|^2

        .. math::
            |\\langle b|\\Phi\\rangle|^2 = \\sum_{nm} c_n^*c_m \\langle 0|V^* U_n^* U_b |0 \\rangle \\langle 0|U_b^* U_m V |0\\rangle

        Args:
            coeff_matrix (np.ndarray): the matrix values of the c_n^* c_m coefficients
            hdmr_values (List): values of the circuit outputs

        Returns:
            float: value of the sum
        """

        if options["use_overlap_test"]:

            # hdmr_values here contains the values of <0|V* Ai* U|0><0|V Aj U|0> for j>=i
            # we first insert these values in a tri up matrix
            size = len(self.matrix_circuits)
            hdmr_matrix = np.zeros((size,size)).astype('complex128')
            hdmr_matrix[np.tril_indices(size)] = hdmr_values

            # add the conj that correspond to the tri low part of the matrix
            # warning the diagonal is also contained in out and we only
            # want to add the conj of the tri up excluding the diag
            hdmr_matrix[np.triu_indices_from(hdmr_matrix, k=1)] = hdmr_matrix[np.tril_indices_from(hdmr_matrix, k=-1)].conj()

            # multiply by the coefficent matrix and sum the values
            out_matrix = coeff_matrix * hdmr_matrix
            out = out_matrix.sum() 
                    
        else:
            # hdmr_values here contains the values of <0|V* Ai* U|0>
            # compute the matrix of the <0|V* Ai* U|0> <0|V Aj U*|0> values
            hdmr_matrix = self.get_coefficient_matrix(hdmr_values)
            out = (coeff_matrix * hdmr_matrix).sum()

        return out

    def _compute_local_terms(
        self,
        coeff_matrix: np.ndarray,
        hdmr_values: np.ndarray,
        norm: float
    ) -> float:
        """Compute the term of the local cost function given by

        .. math::
            \\sum c_i^* c_j \\frac{1}{n} \\sum_n \\langle 0|V^* A_i U Z_n U^* A_j^* V|0\\rangle

        Args:
            coeff_matrix (np.ndarray): the matrix values of the c_n^* c_m coefficients
            hdmr_values (List): values of the circuit outputs

        Returns:
            float: value of the sum
        """
        
        # add all the hadamard test values corresponding to the insertion of Z gates on the same cicuit
        # b_ij = \sum_n \\frac{1}{n} \\sum_n \\langle 0|V^* A_i U Z_n U^* A_j^* V|0\\rangle
        num_zgate = self.matrix_circuits[0].circuit.num_qubits
        hdmr_values = hdmr_values.reshape(-1, num_zgate).mean(1)

        # hdmr_values then contains the values of <0|V* Ai* U|0><0|V Aj U|0> for j>=i
        # we first insert these values in a tri up matrix
        size = len(self.matrix_circuits)
        hdmr_matrix = np.zeros((size,size)).astype('complex128')
        hdmr_matrix[np.triu_indices(size)] = hdmr_values

        # add the conj that correspond to the tri low part of the matrix
        # warning the diagonal is also contained in out and we only
        # want to add the conj of the tri up excluding the diag        
        hdmr_matrix[np.tril_indices_from(hdmr_matrix, k=-1)] = hdmr_matrix[np.triu_indices_from(hdmr_matrix, k=1)].conj()

        
        # multiply by the coefficent matrix and sum the values
        out_matrix = coeff_matrix * hdmr_matrix
        out = (out_matrix).sum()

        # add \sum c_i* cj <0|V Ai* Aj V|0>
        out += norm

        # factor two coming from |0><0| = 1/2(I+Z)
        out /= 2

        return out

    def get_cost_evaluation_function(
        self,
        hdmr_tests_norm: List, 
        hdmr_tests_overlap: List,
        coefficient_matrix: np.ndarray,
        options: Dict
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
        for hdmr in hdmr_tests_norm:
            hdmr.construct_expectation(ansatz_params)

        for hdmr in hdmr_tests_overlap:
            hdmr.construct_expectation(ansatz_params)


        def cost_evaluation(parameters):

            # Create dict associating each parameter with the lists of parameterization values for it
            parameter_sets = np.reshape(parameters, (-1, num_parameters))
            param_bindings = dict(
                zip(ansatz_params, parameter_sets.transpose().tolist())
            )

            # get the sampled output
            hdmr_values_norm = np.array([hdrm.get_value(self._circuit_sampler, param_bindings) 
                                                                for hdrm in hdmr_tests_norm])
            hdmr_values_overlap = np.array([hdrm.get_value(self._circuit_sampler, param_bindings) 
                                                                for hdrm in hdmr_tests_overlap])

            # compute the total cost
            cost = self._assemble_cost_function(hdmr_values_norm, 
                                                hdmr_values_overlap, 
                                                coefficient_matrix,
                                                options)

            # get the intermediate results if required
            if self._callback is not None:
                for param_set in parameter_sets:
                    self._eval_count += 1
                    self._callback(self._eval_count, cost, param_set)
            else:
                self._eval_count += 1
                print(f"VQLS Iteration {self._eval_count} Cost {cost}", end="\r", flush=True) 

            return cost

        return cost_evaluation

 
    def _validate_solve_options(self, options: Union[Dict, None]):
        """validate the options used for the solve methods

        Args:
            options (Union[Dict, None]): options
        """
        valid_keys = self.default_solve_options.keys()

        if options is None:
            options = self.default_solve_options

        else:
            for k in options.keys():
                if k not in self.default_solve_options.keys():
                    raise ValueError("Option {k} not recognized, valid keys are {valid_keys}")
            for k in self.default_solve_options.keys():
                if k not in options.keys():
                    options[k] = self.default_solve_options[k]
        
        if options["use_overlap_test"] and options["use_local_cost_function"]:
            raise ValueError("Local cost function cannot be used with Hadamard Overlap test")

        return options 
        
    def solve(
        self,
        matrix: Union[np.ndarray, QuantumCircuit, List[QuantumCircuit]],
        vector: Union[np.ndarray, QuantumCircuit],
        options: Union[Dict, None],
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

        # validate the options
        options = self._validate_solve_options(options)

        # compute the circuits needed for the hadamard tests
        hdmr_tests_norm, hdmr_tests_overlap = self.construct_circuit(matrix, vector,
                                                                     options)

        # compute he coefficient matrix 
        coefficient_matrix = self.get_coefficient_matrix(np.array([mi.coeff for mi in self.matrix_circuits]))

        # set an expectation for this algorithm run (will be reset to None at the end)
        initial_point = _validate_initial_point(self.initial_point, self.ansatz)
        bounds = _validate_bounds(self.ansatz)

        # Convert the gradient operator into a callable function that is compatible with the
        # optimization routine.
        gradient = self._gradient
        self._eval_count = 0

        # get the cost evaluation function
        cost_evaluation = self.get_cost_evaluation_function(hdmr_tests_norm, 
                                                            hdmr_tests_overlap, 
                                                            coefficient_matrix,
                                                            options)

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


        return solution
