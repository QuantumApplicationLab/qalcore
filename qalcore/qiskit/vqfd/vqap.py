# VariationalVariational Quantum Algorithm based on the minimum potential energy for solving the Poisson equation
# Ref :
# Tutorial :


"""Variational Quantum Algorithm based on the minimum potential energy 
for solving the Poisson equation
Original code : https://github/com/ToyotaCRDL/VQAPoisson
See https://arxiv.org/abs/2106.09333
"""



from ast import operator
from codecs import BOM_UTF16_BE
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


from qiskit.opflow import (
    Z,
    X,
    I,
    StateFn,
    OperatorBase,
    TensoredOp,
    ExpectationBase,
    CircuitSampler,
    ListOp,
    ExpectationFactory,
)

from qiskit.algorithms.optimizers import SLSQP, Minimizer, Optimizer
from qiskit.opflow.gradients import GradientBase
from qalcore.qiskit.vqfd.variational_fd_solver import VariationalFDSolver, VariationalFDSolverResult
from qalcore.qiskit.vqfd.utils import ShiftOperator, ABSuper





class VQAP(VariationalAlgorithm, VariationalFDSolver):

    r"""Systems of linear equations arise naturally in many real-life applications in a wide range
    of areas, such as in the solution of Partial Differential Equations, the calibration of
    financial models, fluid simulation or numerical field calculation. The problem can be defined
    as, given a matrix :math:`A\in\mathbb{C}^{N\times N}` and a vector
    :math:`\vec{b}\in\mathbb{C}^{N}`, find :math:`\vec{x}\in\mathbb{C}^{N}` satisfying
    :math:`A\vec{x}=\vec{b}`.

    Examples:

        .. jupyter-execute:

            from qalcore.qiskit.vqfd.vqap import VQAP
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

        [1] Yuki Sato et al.
        Variational Quantum Algorithm based on the minimum potential energy  for solving the Poisson equation
        `arXiv:2106.09333 <https://arxiv.org/abs/2106.09333>`
    """

    def __init__(
        self,
        ansatz: Optional[QuantumCircuit] = None,
        boundary: Optional[str] = None,
        optimizer: Optional[Union[Optimizer, Minimizer]] = None,
        initial_point: Optional[np.ndarray] = None,
        gradient: Optional[Union[GradientBase, Callable]] = None,
        expectation: Optional[ExpectationBase] = None,
        include_custom: bool = False,
        max_evals_grouped: int = 1,
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
        """
        super().__init__()


        self._num_qubits = None

        self._ansatz = None
        self.ansatz = ansatz

        self._boundary = None
        self.boundary = boundary

        self._optimizer = None
        self.optimizer = optimizer


        self._quantum_instance = None

        if quantum_instance is None:
            quantum_instance = Aer.get_backend("aer_simulator_statevector")
        self.quantum_instance = quantum_instance

        self._callback = None
        self.callback = callback

        tol = 1E-3
        self.tol = {
            'Perdiodic': tol,
            'Neumann': tol,
            'Dirichlet': 0.0
        }[self.boundary]

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
    def boundary(self) -> str:
        """getter for boundary prop

        Returns:
            str: value of boundary
        """
        return self._boundary

    @boundary.setter
    def boundary(self, boundary: Optional[str]):
        """Setter for boundary prop

        Args:
            boundary (Optional[str]): desired boundary

        Raises:
            ValueError: id not recognized
        """
        if boundary is None:
            boundary = 'Dirichlet'
        self._boundary = boundary
        if  self.boundary not in ['Neumann', 'Dirichlet', 'Periodic']:
            raise ValueError('boundary must be Neumann, Dirichlet or Periodic not %s' %self.boundary)


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

    def construct_source_circuit(self, source: Union[np.ndarray, QuantumCircuit] ) -> None:
        """Constructs the different circuits needed to compute the loss function

        Args:
            source (Union[np.ndarray, QuantumCircuit]): the source term of the poisson equation
        """

        # state preparation
        if isinstance(source, QuantumCircuit):
            nb = source.num_qubits
            self.source_circuit = source

        elif isinstance(source, np.ndarray):

            # ensure the vector is double
            source = source.astype("float64")

            # create the circuit
            nb = int(np.log2(len(source)))
            self.source_circuit = QuantumCircuit(nb)

            # prep the vector if its norm is non nul
            vec_norm = np.linalg.norm(source)
            if vec_norm != 0:
                self.source_circuit.prepare_state(source / vec_norm)

    def construct_observable(self) -> None:
        """Constructs all the observable required for the evaluation of the cost function
        """

        # create the obsevable
        zero_op = (I + Z) / 2
        one_op = (I - Z) / 2
        self.observables = {
            "I^(n-1)X" :  TensoredOp((self.num_qubits-1) * [I] ) ^ X,
            "Io^(n-1)X":  TensoredOp((self.num_qubits-1) * [zero_op] ) ^ X,
            "Io^(n-1)I":  TensoredOp((self.num_qubits-1) * [zero_op] ) ^ I,
            "I^(n)O": TensoredOp((self.num_qubits) * [I]) ^ one_op
        }

    def construct_expectation(
        self,
        parameter: Union[List[float], List[Parameter], np.ndarray],
        circuit: QuantumCircuit,
        observable: OperatorBase,
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

        # compose the statefn of the observable on the circuit
        return ~StateFn(observable) @ StateFn(wave_function)

    def construct_shift_ansatz(self):
        """return a circuits that compose the ansatz and the shift operator
        """
        self.shifted_ansatz = self.ansatz.compose(ShiftOperator(self.num_qubits))

    def assemble_circuits(self):
        """Creates a list of circuits/observable/weight required for the calculation
        of the cost function

        Returns:
            Tuple(): circuits, observables, weights
        """

        circuits, observables, weights = [], [], []

        # circuits for the numerator
        circuits += [
            ABSuper(
                self.num_qubits,
                self.source_circuit,
                self.ansatz
            )
        ]
        # observable for the numerator
        observables += [
            self.observables["I^(n)O"]
        ]

        # weiht for numerator
        weights += [1]

        # circuits for denominator
        circuits += [
            self.ansatz,
            self.shifted_ansatz,
        ]

        # circuits for denominator
        observables += [
            self.observables["I^(n-1)X"],
            self.observables["I^(n-1)X"]
        ]

        # weights for denom
        weights += [-1, -1]

        # added terms for different bc
        if self.boundary == 'Dirichlet':
            circuits += [ self.shifted_ansatz ]
            observables += [self.observables["Io^(n-1)X"]]
            weights += [1]

        elif self.boundary == 'Neumann':
            circuits += [ 
                self.shifted_ansatz,
                self.shifted_ansatz]
            observables += [
                self.observables["Io^(n-1)X"],
                self.observables["Io^(n-1)I"]]
                
            weights += [1, -1]

        return circuits, observables, weights

    def process_probability_circuit_output(
        self,
        probability_circuit_output: List,
        weights: List
    ) -> float:
        """Compute the final cost function from the sampled circuit values

        Args:
            probability_circuit_output (List): _description_
            weights (List): _description_

        Returns:
            float: _description_
        """

        numerator = -0.5*probability_circuit_output[0]**2
        denominator = 2.0
        for val, w in zip(probability_circuit_output[1:], weights[1:]):
            denominator += w*val
        
        return numerator/denominator + self.tol
         

    def get_cost_evaluation_function(
        self,
        boundary: str,
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
        circuits, observables, weights = self.assemble_circuits(boundary)
        ansatz_params = self.ansatz.parameters
        expect_ops = []
        for circ, obs in zip(circuits, observables):
            expect_ops.append(self.construct_expectation(ansatz_params, circ, obs))

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
                # out.append(
                #     self.get_probability_from_expected_value(
                #         sampled_expect_op.eval()[0]
                #     )
                # )
                out.append(sampled_expect_op.eval()[0])

            # compute the total cost
            cost = self.process_probability_circuit_output(out, weights)

            # get the internediate results if required
            if self._callback is not None:
                for param_set in parameter_sets:
                    self._eval_count += 1
                    self._callback(self._eval_count, cost, param_set)
            else:
                self._eval_count += 1

            return cost

        return cost_evaluation

    def solve(
        self, 
        source: Union[np.ndarray, QuantumCircuit],
    ) -> VariationalFDSolverResult:

        


        self.construct_source_circuit(source)
        self.construct_observable()


        # get the cost evaluation function
        cost_evaluation = self.get_cost_evaluation_function(self.boundary)

        if callable(self.optimizer):
            opt_result = self.optimizer(  # pylint: disable=not-callable
                fun=cost_evaluation, x0=initial_point, jac=gradient, bounds=bounds
            )
        else:
            opt_result = self.optimizer.minimize(
                fun=cost_evaluation, x0=initial_point, jac=gradient, bounds=bounds
            )

        # create the solution
        solution = VariationalFDSolverResult()

        # optimization data
        solution.optimal_point = opt_result.x
        solution.optimal_parameters = dict(zip(self.ansatz.parameters, opt_result.x))
        solution.optimal_value = opt_result.fun
        solution.cost_function_evals = opt_result.nfev

        # final ansatz
        solution.state = self.ansatz.assign_parameters(solution.optimal_parameters)


        return solution