from qalcore.qiskit.vqls.vqls import VQLS
from qiskit.circuit.library.n_local.real_amplitudes import RealAmplitudes
from qiskit.algorithms.optimizers import COBYLA
from qiskit import Aer, BasicAer
import numpy as np
from qiskit.algorithms.linear_solvers.numpy_linear_solver import NumPyLinearSolver
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt
# from qiskit.primitives import Estimator, Sampler, BackendEstimator
from qiskit.circuit.random import random_circuit

from qiskit_ibm_runtime import QiskitRuntimeService, Estimator, Sampler, Session, Options
from qiskit.utils import QuantumInstance
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qalcore.qiskit.vqls.numpy_unitary_matrices import UnitaryDecomposition

def build_circuit(
        operators,
        apply_initial_state,
    ) -> QuantumCircuit:

    num_qubits = 3
    qc = QuantumCircuit(num_qubits)

    qc.compose(apply_initial_state, list(range(1, num_qubits)), inplace=True)
    qc.h(0)


    # matrix circuit
    for op in operators:

        qc.compose(
            op.control(1),
            qubits=list(range(0, num_qubits)),
            inplace=True,
        )

    # hadamard on ctrl circuit
    qc.h(0)

    return qc



def build_observable():
    """Create the operator to measure |1> on the control qubit.

    Returns:
        Lis[TensoredOp]: List of two observables to measure |1> on the control qubit I^...^I^|1><1|
    """

    num_qubits = 3
    p0 = "I" * num_qubits
    p1 = "I" * (num_qubits-1) + "Z"
    one_op_ctrl = SparsePauliOp([p0,p1], np.array([0.5, -0.5]))
    return one_op_ctrl


# define the problem
A = np.random.rand(4, 4)
A = A + A.T
b = np.random.rand(4)

# define the circuits locally
vector = b.astype("float64")
vec_norm = np.linalg.norm(vector)
vector_circuit = QuantumCircuit(2)
vector_circuit.prepare_state(vector / vec_norm)

# define the matrix circuits
matrix_circuits = UnitaryDecomposition(matrix=A)



# define ansatz
ansatz = RealAmplitudes(2, entanglement="full", reps=3, insert_barriers=False)

#get circuit
hdmr = build_circuit([matrix_circuits[0].circuit.inverse(), matrix_circuits[1].circuit],ansatz)

# get observable
obs = build_observable()

# define the runtime
service = QiskitRuntimeService()
backend = "ibmq_qasm_simulator"
backend =  "simulator_statevector"

with Session(service=service, backend=backend) as session:

    options = Options()
    # options.optimization_level = 3
    estimator = Estimator(session=session, options=options)

    # Sampler = Sampler(session=session, backend=backend)
    # vqls = VQLS(
    #     estimator,
    #     ansatz,
    #     COBYLA(maxiter=200, disp=True)
    # )

    # opt= {"use_overlap_test": False,
    #     "use_local_cost_function": False}
    # hdmr_tests_norm, hdmr_tests_overlap = vqls.construct_circuit(A, b, opt)

    parameters = np.random.rand(ansatz.num_parameters)
    # hdmr_values_norm = np.array([hdrm.get_value(vqls.estimator, parameters) for hdrm in hdmr_tests_norm])

    # qc = QuantumCircuit(3)
    # qc.h(0)
    # qc.compose(vqls.ansatz,[1,2], inplace=True)

    job = estimator.run(hdmr, obs, parameters)
    job.result()
    # res = vqls.solve(A, b, opt)


# ref_solution = classical_solution.state / np.linalg.norm(classical_solution.state)
# vqls_solution = np.real(Statevector(res.state).data)


# plt.scatter(ref_solution, vqls_solution)
# plt.plot([-1, 1], [-1, 1], "--")
# plt.show()