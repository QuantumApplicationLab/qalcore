from qalcore.qiskit.vqls.vqls import VQLS
from qiskit.circuit.library.n_local.real_amplitudes import RealAmplitudes
from qiskit.algorithms.optimizers import COBYLA
from qiskit import Aer
import numpy as np
from qiskit.algorithms.linear_solvers.numpy_linear_solver import NumPyLinearSolver



A = np.random.rand(4, 4)
A = A + A.T

b = np.random.rand(4)

classical_solution = NumPyLinearSolver().solve(A, b / np.linalg.norm(b))

ansatz = RealAmplitudes(2, entanglement="full", reps=3, insert_barriers=False)

vqls = VQLS(
    ansatz=ansatz,
    optimizer=COBYLA(maxiter=200, disp=True),
    quantum_instance=Aer.get_backend("aer_simulator_statevector"),
    use_overlap_test=False,
    use_local_cost_function=True   
)

res = vqls.solve(A, b)