# Grab functions and modules from dependencies

from abc import ABC, abstractmethod
from typing import Optional, Union, List, Callable, Tuple

from types import SimpleNamespace
import numpy as np

import scipy.linalg as spla
import scipy.optimize as opt
from scipy.optimize import OptimizeResult
import mthree

# Grab functions and modules from Qiskit needed
from qiskit import QuantumCircuit, transpile
from qiskit import Aer

from qiskit.providers import Backend

from qiskit.circuit import Parameter
from qiskit.circuit.library.n_local.real_amplitudes import RealAmplitudes


from qiskit.quantum_info import Statevector
from qiskit.quantum_info import Operator
from qiskit.quantum_info.operators.base_operator import BaseOperator

from qiskit.algorithms.variational_algorithm import VariationalAlgorithm
from qiskit.algorithms.optimizers import SLSQP, Minimizer, Optimizer
from qiskit.algorithms.variational_algorithm import VariationalResult
from qiskit.algorithms.linear_solvers.linear_solver import LinearSolverResult
from qiskit.algorithms.linear_solvers.observables.linear_system_observable import (
    LinearSystemObservable,
)
from qiskit.algorithms.minimum_eigen_solvers.vqe import (
    _validate_bounds,
    _validate_initial_point,
)


from qiskit.utils import QuantumInstance
from qiskit.utils.backend_utils import is_aer_provider, is_statevector_backend
from qiskit.utils.validation import validate_min


from qiskit.opflow.gradients import GradientBase
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




# The entrypoint for our Runtime Program
def main(
    backend,
    user_messenger,
    matrix,
    rhs,
    ansatz,
    x0=None,
    optimizer="SPSA",
    optimizer_config={"maxiter": 100},
    shots=8192,
    use_measurement_mitigation=False,
):

    """
    The main sample VQLS program.

    Parameters:
        backend (ProgramBackend): Qiskit backend instance.
        user_messenger (UserMessenger): Used to communicate with the
                                        program user.
        matrix (np.ndarray): the matrix of the linear system.
        rhs (np.ndarray): the right habd side of the linear system.
        ansatz (QuantumCircuit): the quantum circuit of the ansatz.

        x0 (array_like): Optional, initial vector of parameters.
        optimizer (str): Optional, string specifying classical optimizer,
                         default='SPSA'.
        optimizer_config (dict): Optional, configuration parameters for the
                                 optimizer.
        shots (int): Optional, number of shots to take per circuit.
        use_measurement_mitigation (bool): Optional, use measurement mitigation,
                                           default=False.

    Returns:
        VariationalLinearSolverResult: The result of the optimization of the linear system
    """

    vqls = VQLS(ansatz=ansatz, optimizer=optimizer, quantum_instance=backend)

    use_state_vector = is_statevector_backend(backend)
    if use_state_vector:
        print("Use exact statistic from state vector")
        full_circs = vqls.construct_circuit(
            matrix, rhs, appply_explicit_measurement=False
        )
    else:
        print("Use count statistic")
        full_circs = vqls.construct_circuit(
            matrix, rhs, appply_explicit_measurement=True
        )
        meas_strings = ["1"] * len(full_circs)

    # Get the number of parameters in the ansatz circuit.
    num_params = vqls.ansatz.num_parameters

    # Use a given initial state, if any, or do random initial state.
    if x0:
        x0 = np.asarray(x0, dtype=float)
        if x0.shape[0] != num_params:
            raise ValueError(
                "Number of params in x0 ({}) does not match number \
                              of ansatz parameters ({})".format(
                    x0.shape[0], num_params
                )
            )
    else:
        x0 = 2 * np.pi * np.random.rand(num_params)

    # Because we are in general targeting a real quantum system, our circuits must be transpiled
    # to match the system topology and, hopefully, optimize them.
    # Here we will set the transpiler to the most optimal settings where 'sabre' layout and
    # routing are used, along with full O3 optimization.

    # This works around a bug in Qiskit where Sabre routing fails for simulators (Issue #7098)
    trans_dict = {}
    if not backend.configuration().simulator:
        trans_dict = {"layout_method": "sabre", "routing_method": "sabre"}
    trans_circs = transpile(full_circs, backend, optimization_level=3, **trans_dict)

    # If using measurement mitigation we need to find out which physical qubits our transpiled
    # circuits actually measure, construct a mitigation object targeting our backend, and
    # finally calibrate our mitgation by running calibration circuits on the backend.
    if use_measurement_mitigation:
        maps = mthree.utils.final_measurement_mapping(trans_circs)
        mit = mthree.M3Mitigation(backend)
        mit.cals_from_system(maps)

    # Here we define a callback function that will stream the optimizer parameter vector
    # back to the user after each iteration.  This uses the `user_messenger` object.
    # Here we convert to a list so that the return is user readable locally, but
    # this is not required.
    def callback(xk):
        if user_messenger is not None:
            user_messenger.publish(list(xk))

    # This is the primary VQE function executed by the optimizer. This function takes the
    # parameter vector as input and returns the energy evaluated using an ansatz circuit
    # bound with those parameters.
    def vqls_func(params):

        # Attach (bind) parameters in params vector to the transpiled circuits.
        bound_circs = [circ.bind_parameters(params) for circ in trans_circs]

        if use_state_vector:
            state_vector = [Statevector(circ).data for circ in bound_circs]
            probas = [vqls.get_probability_from_statevector(sv) for sv in state_vector]

        else:

            # Submit the job and get the resultant counts back
            counts = backend.run(bound_circs, shots=shots).result().get_counts()

            # If using measurement mitigation apply the correction and
            # compute expectation values from the resultant quasiprobabilities
            # using the measurement strings.
            if use_measurement_mitigation:
                quasi_collection = mit.apply_correction(counts, maps)
                expvals = quasi_collection.expval(meas_strings)
            # If not doing any mitigation just compute expectation values
            # from the raw counts using the measurement strings.
            # Since Qiskit does not have such functionality we use the convenence
            # function from the mthree mitigation module.
            else:
                expvals = mthree.utils.expval(counts, meas_strings)

            # get the probas
            probas = [vqls.get_probability_from_expected_value(e) for e in expvals]

        # The final cost is obatined via the vqls process_probability_circuit_output
        # method
        cost = vqls.process_probability_circuit_output(probas)
        return cost

    # Here is where we actually perform the computation.  We begin by seeing what
    # optimization routine the user has requested, eg. SPSA verses SciPy ones,
    # and dispatch to the correct optimizer.  The selected optimizer starts at
    # x0 and calls 'vqe_func' everytime the optimizer needs to evaluate the cost
    # function.  The result is returned as a SciPy OptimizerResult object.
    # Additionally, after every iteration, we use the 'callback' function to
    # publish the interim results back to the user. This is important to do
    # so that if the Program terminates unexpectedly, the user can start where they
    # left off.

    # Since SPSA is not in SciPy need if statement
    if optimizer == "SPSA":
        opt_result = fmin_spsa(
            vqls_func, x0, args=(), **optimizer_config, callback=callback
        )

    # All other SciPy optimizers here
    else:
        opt_result = opt.minimize(
            vqls_func,
            x0,
            method=optimizer,
            options=optimizer_config,
            tol=1e-3,
            callback=callback,
        )

        opt_result = OptimizeResult(
            fun=vqls_func(opt_result.x),
            x=opt_result.x,
            nit=0,
            nfev=0,
            message="Optimization terminated successfully.",
            success=True,
        )
    return opt_result


def fmin_spsa(
    func,
    x0,
    args=(),
    maxiter=100,
    a=1.0,
    alpha=0.602,
    c=1.0,
    gamma=0.101,
    callback=None,
):
    """
    Minimization of scalar function of one or more variables using simultaneous
    perturbation stochastic approximation (SPSA).

    Parameters:
        func (callable): The objective function to be minimized.

                          ``fun(x, *args) -> float``

                          where x is an 1-D array with shape (n,) and args is a
                          tuple of the fixed parameters needed to completely
                          specify the function.

        x0 (ndarray): Initial guess. Array of real elements of size (n,),
                      where ‘n’ is the number of independent variables.

        maxiter (int): Maximum number of iterations.  The number of function
                       evaluations is twice as many. Optional.

        a (float): SPSA gradient scaling parameter. Optional.

        alpha (float): SPSA gradient scaling exponent. Optional.

        c (float):  SPSA step size scaling parameter. Optional.

        gamma (float): SPSA step size scaling exponent. Optional.

        callback (callable): Function that accepts the current parameter vector
                             as input.

    Returns:
        OptimizeResult: Solution in SciPy Optimization format.

    Notes:
        See the `SPSA homepage <https://www.jhuapl.edu/SPSA/>`_ for usage and
        additional extentions to the basic version implimented here.
    """
    A = 0.01 * maxiter
    x0 = np.asarray(x0)
    x = x0

    for kk in range(maxiter):
        ak = a * (kk + 1.0 + A) ** -alpha
        ck = c * (kk + 1.0) ** -gamma
        # Bernoulli distribution for randoms
        deltak = 2 * np.random.randint(2, size=x.shape[0]) - 1
        grad = (func(x + ck * deltak, *args) - func(x - ck * deltak, *args)) / (
            2 * ck * deltak
        )
        x -= ak * grad

        if callback is not None:
            callback(x)

    return OptimizeResult(
        fun=func(x, *args),
        x=x,
        nit=maxiter,
        nfev=2 * maxiter,
        message="Optimization terminated successfully.",
        success=True,
    )


class VariationalLinearSolverResult(VariationalResult):
    """A base class for linear systems results using variational methods

    The  linear systems variational algorithms return an object of the type ``VariationalLinearSystemsResult``
    with the information about the solution obtained.
    """

    def __init__(self) -> None:
        super().__init__()

    @property
    def cost_function_evals(self) -> Optional[int]:
        """Returns number of cost optimizer evaluations"""
        return self._cost_function_evals

    @cost_function_evals.setter
    def cost_function_evals(self, value: int) -> None:
        """Sets number of cost function evaluations"""
        self._cost_function_evals = value

    @property
    def state(self) -> Union[QuantumCircuit, np.ndarray]:
        """return either the circuit that prepares the solution or the solution as a vector"""
        return self._state

    @state.setter
    def state(self, state: Union[QuantumCircuit, np.ndarray]) -> None:
        """Set the solution state as either the circuit that prepares it or as a vector.

        Args:
            state: The new solution state.
        """
        self._state = state


class VariationalLinearSolver(ABC):
    """An abstract class for linear system solvers in Qiskit."""

    @abstractmethod
    def solve(
        self,
        matrix: Union[np.ndarray, QuantumCircuit],
        vector: Union[np.ndarray, QuantumCircuit],
        observable: Optional[
            Union[
                BaseOperator,
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
        """Solve the system and compute the observable(s)

        Args:
            matrix: The matrix specifying the system, i.e. A in Ax=b.
            vector: The vector specifying the right hand side of the equation in Ax=b.
            observable: Optional information to be extracted from the solution.
                Default is the probability of success of the algorithm.
            observable_circuit: Optional circuit to be applied to the solution to extract
                information. Default is ``None``.
            post_processing: Optional function to compute the value of the observable.
                Default is the raw value of measuring the observable.

        Returns:
            The result of the linear system.
        """
        raise NotImplementedError


class UnitaryDecomposition:
    r"""Compute the unitary decomposition of a general matrix
    See:
        https://math.stackexchange.com/questions/1710247/every-matrix-can-be-written-as-a-sum-of-unitary-matrices/1710390#1710390
    """

    def __init__(
        self,
        matrix: Optional[np.ndarray] = None,
        circuits: Optional[Union[QuantumCircuit, List[QuantumCircuit]]] = None,
        coefficients: Optional[
            Union[float, complex, List[float], List[complex]]
        ] = None,
        check_decomposition: Optional[bool] = True,
        normalize_coefficients: Optional[bool] = True,
    ):
        """Unitary decomposition

        Args:
            matrix (Optional[np.ndarray], optional): input matrix to be transformed.
            circuit (Optional[Union[QuantumCircuit, List[QuantumCircuit]]], optional): quantum circuit(s) representing the matrix.
            coefficients (Optional[Union[float, complex, List[float], List[complex]]], optional): coefficients of associated with the input quantum circuits.
            check_decomposition (Optional[bool], optional): Check if the decomposition matches the input matrix. Defaults to True.
            normalize_coefficients (Optional[bool], optional): normalize the coefficients of the decomposition. Defaults to True.
        """

        self._matrix = None
        self.matrix = matrix

        self._circuits = None
        self.circuits = circuits

        self._coefficients = None
        self.coefficients = coefficients

        self._unitary_matrices = None

        self.iiter = None

        # if circuits are provided
        if self._circuits is not None:

            # case where the cioefficients are not provided
            if self._coefficients is None:
                if len(self._circuits) == 1:
                    self.coefficients = [1.0]
                else:
                    raise ValueError(
                        "Value of coefficients must be provided for multiple circuits"
                    )

            # check that we have same number of coefficients and circuits
            if len(self._circuits) != len(self._coefficients):
                raise ValueError(
                    "different number of coefficients and circuits provided as input"
                )

            # set the number of qubits and checkthe size of all circuits
            self.num_qubits = self._circuits[0].num_qubits
            for qc in self._circuits:
                if qc.num_qubits != self.num_qubits:
                    raise ValueError("All circuits must have the same number of qubits")

            self._unitary_matrices = [Operator(qc).data for qc in self._circuits]

        # if a matrix is provided
        elif self._matrix is not None:

            if self._circuits is not None:
                raise ValueError(
                    "Circuits cannot be provided if matrix is provided as input"
                )

            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Input matrix must be square!")

            if np.log2(matrix.shape[0]) % 1 != 0:
                raise ValueError("Input matrix dimension must be 2^n!")

            self.coefficients, self.unitary_matrices = self.decompose_numpy_matrix(
                check=check_decomposition, normalize_coefficients=normalize_coefficients
            )

            self.num_qubits = int(np.log2(matrix.shape[0]))
            self.circuits = self.create_circuits(self.unitary_matrices)

        self.num_circuits = len(self._circuits)

    @property
    def matrix(self) -> np.ndarray:
        """return the matrix of the decomposition."""
        if self._matrix is None:
            self._matrix = self.recompose(self.coefficients, self.unitary_matrices)
        return self._matrix

    @matrix.setter
    def matrix(self, matrix: np.ndarray) -> None:
        """Sets the matrix"""
        self._matrix = matrix

    @property
    def circuits(self) -> List[QuantumCircuit]:
        """return the circuits of the decomposition."""
        return self._circuits

    @circuits.setter
    def circuits(self, circuits: Union[QuantumCircuit, List[QuantumCircuit]]) -> None:
        """Sets the matrix"""
        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]
        self._circuits = circuits

    @property
    def coefficients(self) -> Union[List[float], List[complex]]:
        """return the coefficients of the decomposition."""
        return self._coefficients

    @coefficients.setter
    def coefficients(
        self, coefficients: Union[float, complex, List[float], List[complex]]
    ) -> None:
        """Sets the matrix"""
        if not isinstance(coefficients, List):
            coefficients = [coefficients]
        self._coefficients = [c for c in np.array(coefficients).astype(np.cdouble)]

    @property
    def num_qubits(self) -> int:
        """return the numner of qubits"""
        return self._num_qubits

    @num_qubits.setter
    def num_qubits(self, num_qubits: int) -> None:
        """Set the number of qubits"""
        self._num_qubits = num_qubits

    @property
    def unitary_matrices(self) -> int:
        """return the unitary matrices"""
        return self._unitary_matrices

    @unitary_matrices.setter
    def unitary_matrices(
        self, unitary_matrices: Union[np.ndarray, List[np.ndarray]]
    ) -> None:
        """Set the number of qubits"""
        if isinstance(unitary_matrices, np.ndarray):
            unitary_matrices = [unitary_matrices]
        self._unitary_matrices = unitary_matrices

    def __iter__(self):
        self.iiter = 0
        return self

    def __next__(self):
        if self.iiter < self.num_circuits:
            out = SimpleNamespace(
                coeff=self._coefficients[self.iiter], circuit=self._circuits[self.iiter]
            )
            self.iiter += 1
            return out
        raise StopIteration

    def __len__(self):
        return len(self._circuits)

    def __getitem__(self, index):
        return SimpleNamespace(
            coeff=self._coefficients[index], circuit=self._circuits[index]
        )

    @staticmethod
    def get_auxilliary_matrix(x: np.ndarray) -> np.ndarray:
        """Compute i * sqrt(I - x^2)

        Args:
            x (np.ndarray): input matrix

        Returns:
            np.ndarray: values of i * sqrt(I - x^2)
        """
        return 1.0j * spla.sqrtm(np.eye(len(x)) - x @ x)

    def decompose_numpy_matrix(
        self,
        check: Optional[bool] = False,
        normalize_coefficients: Optional[bool] = False,
    ) -> Tuple[List[float], List[np.ndarray]]:
        """Decompose a generic numpy matrix into a sum of unitary matrices

        Args:
            check (Optional[bool], optional): _description_. Defaults to False.
            normalize_coefficients (Optional[bool], optional): _description_. Defaults to False.

        Returns:
            Tuple: list of coefficients and numpy matrix of the decompostion
        """

        # Normalize
        norm = np.linalg.norm(self._matrix)
        mat = self._matrix / norm

        mat_real = np.real(mat)
        mat_imag = np.imag(mat)

        coef_real = norm * 0.5
        coef_imag = coef_real * 1j

        ## Get the matrices
        unitary_matrices, unitary_coefficients = [], []
        if not np.allclose(mat_real, 0.0):
            aux_mat = self.get_auxilliary_matrix(mat_real)
            unitary_matrices += [mat_real + aux_mat, mat_real - aux_mat]
            unitary_coefficients += [coef_real] * 2

        if not np.allclose(mat_imag, 0.0):
            aux_mat = self.get_auxilliary_matrix(mat_imag)
            unitary_matrices += [mat_imag + aux_mat, mat_imag - aux_mat]
            unitary_coefficients += [coef_imag] * 2

        if check:
            mat_recomp = self.recompose(unitary_coefficients, unitary_matrices)
            assert np.allclose(self._matrix, mat_recomp)

        if normalize_coefficients:
            unitary_coefficients = self.normalize_coefficients(unitary_coefficients)

        return unitary_coefficients, unitary_matrices

    def normalize_coefficients(self, unit_coeffs: List[float]) -> List[float]:
        """Normalize the coefficients

        Args:
            unit_coeffs (List[float]): list of coefficients

        Returns:
            List[float]: List of normalized coefficients
        """
        sum_coeff = np.array(unit_coeffs).sum()
        return [u / sum_coeff for u in unit_coeffs]

    def recompose(
        self, unit_coeffs: List[float], unit_mats: List[np.ndarray]
    ) -> np.ndarray:
        """Rebuilds the original matrix from the decomposed one.

        Args:
            unit_coeffs (List[float]): coefficients of the decomposition
            unit_mats (List[np.ndarray]): matrices of the decomposition

        Returns:
            np.ndarray: recomposed matrix
        """
        recomp = np.zeros_like(unit_mats[0])
        for c, m in zip(unit_coeffs, unit_mats):
            recomp += c * m
        return recomp

    def create_circuits(self, unit_mats: List[np.ndarray]) -> List[QuantumCircuit]:
        """Contstruct the quantum circuits.

        Args:
            unit_mats (List[np.ndarray]): list of unitary matrices of the decomposition.

        Returns:
            List[QuantumCircuit]: list of resulting quantum circuits.
        """
        circuits = []
        for m in unit_mats:
            qc = QuantumCircuit(self.num_qubits)
            qc.unitary(m, qc.qubits)
            circuits.append(qc)
        return circuits


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
        # these are the coefficients if the qubits of register A and B
        # are ordered as A0 B0 A1 B1 .... AN BN
        c0 = np.array([1,1,1,-1])
        coeffs = np.array([1,1,1,-1])
        for _ in range(1,self.operator_num_qubits):
            coeffs = np.tensordot(coeffs, c0, axes=0).flatten()


        # create all the possible bit strings of a single register
        bit_strings = []
        for i in range(2**(self.operator_num_qubits)):
            bit_strings.append( f"{i:b}".zfill(self.operator_num_qubits) )

        # coeff in the A0 A1 .. AN B0 B1 ... BN
        reordered_coeffs = np.zeros_like(coeffs)

        # Reorder the coefficients from 
        # A0 B0 A1 B1 ... AN BN => A0 A1 .. AN B0 B1 ... BN
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
# Variational Quantum Linear Solver
# Ref :
# Tutorial :


"""Variational Quantum Linear Solver

See https://arxiv.org/abs/1909.05820
"""



from typing import Optional, Union, List, Callable, Tuple
import numpy as np
import itertools

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

# from qiskit.algorithms.linear_solvers.observables.linear_system_observable import (
#     LinearSystemObservable,
# )

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

        # create only the circuit for <psi|psi> =  <0|V A_n ^* A_m V|0>
        # with n != m as the diagonal terms (n==m) always give a proba of 1.0
        hdmr_tests_norm = self._get_norm_circuits(appply_explicit_measurement)
        
        # create the circuits for <b|psi> 
        # local cost function
        if self._use_local_cost_function:
            hdmr_tests_overlap = self._get_local_circuits(appply_explicit_measurement)
        # global cost function 
        else:
            hdmr_tests_overlap = self._get_global_circuits(appply_explicit_measurement)

        return hdmr_tests_norm, hdmr_tests_overlap

    def _get_norm_circuits(self, appply_explicit_measurement: bool) -> List[QuantumCircuit]:
        """_summary_

        Raises:
            RuntimeError: _description_

        Returns:
            List[QuantumCircuit]: _description_
        """

        hdmr_tests_norm = []

        for ii in range(len(self.matrix_circuits)):
            mi = self.matrix_circuits[ii]

            for jj in range(ii + 1, len(self.matrix_circuits)):
                mj = self.matrix_circuits[jj]
                hdmr_tests_norm.append( HadammardTest(
                    operators=[mi.circuit.inverse(), mj.circuit],
                    apply_initial_state=self._ansatz,
                    apply_measurement=appply_explicit_measurement,
                    )
                )
        return hdmr_tests_norm

    def _get_local_circuits(self, appply_explicit_measurement: bool) -> List[QuantumCircuit]:
        """_summary_

        Args:
            appply_explicit_measurement (bool): _description_

        Returns:
            List[QuantumCircuit]: _description_
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
                        apply_measurement = appply_explicit_measurement 
                        )
                    )
        return hdmr_tests_overlap

    def _get_global_circuits(self, appply_explicit_measurement: bool) -> List[QuantumCircuit]:
        """_summary_

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
        if self._use_overlap_test:

            for ii in range(len(self.matrix_circuits)):
                mi = self.matrix_circuits[ii]

                for jj in range(ii, len(self.matrix_circuits)):
                    mj = self.matrix_circuits[jj]

                    hdmr_tests_overlap.append(HadammardOverlapTest(
                        operators = [self.vector_circuit, mi.circuit, mj.circuit],
                        apply_initial_state = self.ansatz,
                        apply_measurement = appply_explicit_measurement 
                        )
                    )
        else:
            
            for mi in self.matrix_circuits:
                hdmr_tests_overlap.append(HadammardTest(
                    operators=[self.ansatz, mi.circuit, self.vector_circuit.inverse()],
                    apply_measurement=appply_explicit_measurement,
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
        coefficient_matrix: np.ndarray
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

        if self._use_local_cost_function:
            # compute all terms in 
            # \sum c_i* c_j 1/n \sum_n <0|V* Ai U Zn U* Aj* V|0>
            sum_terms = self._compute_local_terms(
                coefficient_matrix, hdmr_values_overlap, norm
            )

        else:
            # compute all the terms in 
            # |<b|\phi>|^2 = \sum c_i* cj <0|U* Ai V|0><0|V* Aj* U|0>
            sum_terms = self._compute_global_terms(
                coefficient_matrix, hdmr_values_overlap
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

        if self._use_overlap_test:

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
                                                coefficient_matrix)

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

    def _calculate_observable(
        self,
        solution: QuantumCircuit,
        observable: Optional[BaseOperator] = None,
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
                BaseOperator,
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

        # compute the circuits needed for the hadamard tests
        hdmr_tests_norm, hdmr_tests_overlap = self.construct_circuit(matrix, vector)

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
        cost_evaluation = self.get_cost_evaluation_function(hdmr_tests_norm, hdmr_tests_overlap, coefficient_matrix)

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

